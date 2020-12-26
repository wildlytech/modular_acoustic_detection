"""
Training a Multilabel Model
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from tensorflow.compat.v1.keras.models import Sequential,model_from_json
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.optimizers import Adam
from youtube_audioset import get_recursive_sound_names
import balancing_dataset
import os
import json
from colorama import Fore,Style
import argparse
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt
#########################################################
    #Description and Help
#########################################################
DESCRIPTION = "Reads config file from user and trains multilabel model accordingly."
HELP = "Input config filepath for the required model to be trained."
#########################################################
    #Parse the arguments
#########################################################
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("-cfg_json",action="store",help=HELP,required=True)
result = parser.parse_args()
cfg_path = result.cfg_json

def read_config(filepath):
    with open(filepath) as f:
        config = json.load(f)
    return config

config = read_config(cfg_path)


output_wt_file = config["train"]["outputWeightFile"]
pathToFileDirectory = "/".join(output_wt_file.split('/')[:-1]) + '/'
if not os.path.exists(pathToFileDirectory):
    os.makedirs(pathToFileDirectory)

assert(config["ontology"]["useYoutubeAudioSet"])
# If single file or null, then convert to list
ontologyExtFiles = config["ontology"]["extension"]

if ontologyExtFiles is None:
    ontologyExtFiles = []
elif type(ontologyExtFiles) != list:
    ontologyExtFiles = [ontologyExtFiles]


pos_sounds = {}
neg_sounds = {}
for label_dicts in config["labels"]:
    lab_name = label_dicts["aggregatePositiveLabelName"]
    comprising_pos_labels = get_recursive_sound_names(label_dicts["positiveLabels"],"./",ontologyExtFiles)
    pos_sounds[lab_name] = comprising_pos_labels
    if label_dicts["negativeLabels"]!=None:
        neg_lab_name = label_dicts["aggregateNegativeLabelName"]
        comprising_neg_labels = get_recursive_sound_names(label_dicts["negativeLabels"],"./",ontologyExtFiles)
        comprising_neg_labels = comprising_neg_labels.difference(comprising_pos_labels)
        neg_sounds[neg_lab_name] = comprising_neg_labels




########################################################################
      # Importing balanced data from the function.
      # Including audiomoth annotated files for training
########################################################################

def subsample_dataframe(dataframe, subsample):
    """
    Subsample examples from the dataframe
    """
    if subsample is not None:
        if subsample > 0:
            # If subsample is less than size of dataframe, then
            # don't allow replacement when sampling
            # Otherwise, the intention is to oversample
            dataframe = dataframe.sample(subsample, replace=(subsample > dataframe.shape[0]))
        else:
            dataframe = pd.DataFrame([], columns=dataframe.columns)

    return dataframe

def split_and_subsample_dataframe(dataframe, validation_split, subsample):
    """
    Perform validation split and sub/over sampling of dataframe
    Returns train and test dataframe
    """

    # split before sub/oversampling to ensure there is no leakage between train and test sets
    test_size = int(validation_split*dataframe.shape[0])

    train_size = dataframe.shape[0] - test_size

    if test_size == 0:
        train_df = dataframe
        test_df = pd.DataFrame({}, columns=dataframe.columns)
    elif train_size == 0:
        train_df = pd.DataFrame({}, columns=dataframe.columns)
        test_df = dataframe
    else:
        train_df, test_df = train_test_split(dataframe,
                                             test_size=test_size,
                                             random_state=42)

    test_subsample = int(subsample*validation_split)
    train_subsample = subsample - test_subsample

    if (train_df.shape[0] == 0) and (train_subsample > 0):
        print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)

    else:
        train_df = subsample_dataframe(train_df, train_subsample)

    if (test_df.shape[0] == 0) and (test_subsample > 0):
        print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)

    else:
        test_df = subsample_dataframe(test_df, test_subsample)

    return train_df, test_df

def get_select_vector(dataframe, label_filter_arr):
      """
      Get the boolean select vector on the dataframe from the label filter
      """
      return dataframe['labels_name'].apply(lambda arr: np.any([x.lower() in label_filter_arr for x in arr]))

def import_dataframes(dataframe_file_list,
                      positive_label_filter_arr,
                      negative_label_filter_arr,
                      validation_split):
    """
    Iterate through each pickle file and import a subset of the dataframe
    """
    # All entries that have a pattern path need special handling
    # Specifically, all the files that match the pattern path
    # need to be determined
    pattern_file_dicts = [x for x in dataframe_file_list if "patternPath" in list(x.keys())]

    # Keep just the entries that don't have a pattern path
    # We'll add entries for each pattern path separately
    dataframe_file_list = [x for x in dataframe_file_list if "patternPath" not in list(x.keys())]

    # Expand out each pattern path entry into individual ones with fixed paths
    for input_file_dict in pattern_file_dicts:
        pattern_path = input_file_dict["patternPath"]

        # Get list of any exclusions that should be made
        if "excludePaths" in list(input_file_dict.keys()):
            exclude_paths = input_file_dict["excludePaths"]

            if exclude_paths is None:
                exclude_paths = []
            elif type(exclude_paths) != list:
                exclude_paths = [exclude_paths]
        else:
            exclude_paths = []

        # Make excluded paths a set for fast lookup
        exclude_paths = set(exclude_paths)

        # Search for all paths that match this pattern
        search_results = glob(pattern_path)

        # If path is in the excluded paths, then ignore it
        search_results = [x for x in search_results if x not in exclude_paths]

        if len(search_results) == 0:
            print(Fore.RED, "No file matches pattern criteria:", pattern_path, "Excluded Paths:", exclude_paths,Style.RESET_ALL)
            assert (False)

        for path in search_results:
            # Add an entry with fixed path
            fixed_path_dict = input_file_dict.copy()

            # Remove keys that are used for pattern path entry
            # Most other keys should be the same as the pattern path entry
            del fixed_path_dict["patternPath"]
            if "excludePaths" in list(fixed_path_dict.keys()):
                del fixed_path_dict["excludePaths"]

            # Make it look like a fixed path entry
            fixed_path_dict["path"] = path
            dataframe_file_list += [fixed_path_dict]

    # Proceed with importing all dataframe files
    list_of_train_dataframes = []
    list_of_test_dataframes = []
    for input_file_dict in dataframe_file_list:

        assert("patternPath" not in list(input_file_dict.keys()))
        assert("path" in list(input_file_dict.keys()))

        print("Importing", input_file_dict["path"], "...")

        with open(input_file_dict["path"], 'rb') as file_obj:

            # Load the file

            df = pickle.load(file_obj)

        # Filtering the sounds that are exactly 10 seconds
        # Examples should be exactly 10 seconds. Anything else
        # is not a valid input to the model

        df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]

        final_dfs = []

        '''for key in positive_label_filter_arr.keys():

            positive_filter = positive_label_filter_arr[key]
            if key in negative_label_filter_arr.keys():
                negative_filter = negative_label_filter_arr[key]
            else:
                negative_filter = None
            # Only use examples that have a label in label filter array

            positive_example_select_vector = get_select_vector(df, positive_filter)
            positive_examples_df = df.loc[positive_example_select_vector]

            # This ensures there no overlap between positive and negative examples
            negative_example_select_vector = ~positive_example_select_vector
            if negative_filter is not None:
                # Exclude even further examples that don't fall into the negative label filter
                negative_example_select_vector &= get_select_vector(df, negative_filter)
            negative_examples_df = df.loc[negative_example_select_vector]

            # No longer need df after this point

            train_positive_examples_df, test_positive_examples_df = \
                    split_and_subsample_dataframe(dataframe=positive_examples_df,
                                                  validation_split=validation_split,
                                                  subsample=input_file_dict["subsample"])

            del positive_examples_df

            train_negative_examples_df, test_negative_examples_df = \
                    split_and_subsample_dataframe(dataframe=negative_examples_df,
                                                  validation_split=validation_split,
                                                  subsample=input_file_dict["subsample"])

            del negative_examples_df

            # append to overall list of examples

            list_of_train_dataframes += [train_positive_examples_df, train_negative_examples_df]



            list_of_test_dataframes += [test_positive_examples_df, test_negative_examples_df]


        train_df = pd.concat(list_of_train_dataframes, ignore_index=True)
        final_dfs_train.append(train_df)
        test_df = pd.concat(list_of_test_dataframes, ignore_index=True)
        final_dfs_test.append(test_df)'''
        final_dfs.append(df)

    DF = pd.concat(final_dfs,ignore_index=True)

    DF_TRAIN,DF_TEST = split_and_subsample_dataframe(dataframe=DF,
                                                     validation_split=validation_split,
                                                     subsample=input_file_dict["subsample"])

    #DF_TRAIN = pd.concat(final_dfs_train, ignore_index=True)
    #DF_TEST = pd.concat(final_dfs_test, ignore_index=True)
    print("Import done.")

    return DF_TRAIN, DF_TEST


DF_TRAIN, DF_TEST = import_dataframes(dataframe_file_list=config["train"]["inputDataFrames"],
                              positive_label_filter_arr=pos_sounds,
                              negative_label_filter_arr=neg_sounds,
                              validation_split=config["train"]["validationSplit"])


wavefiles = DF_TRAIN.wav_file
labels = DF_TRAIN.labels_name



LABELS_BINARIZED_TRAIN = pd.DataFrame()
LABELS_BINARIZED_TEST = pd.DataFrame()
for key in pos_sounds.keys():
    FULL_NAME = key
    POSITIVE_LABELS = pos_sounds[key]
    LABELS_BINARIZED_TRAIN[FULL_NAME] = 1.0 * get_select_vector(DF_TRAIN, POSITIVE_LABELS)


    LABELS_BINARIZED_TEST[FULL_NAME] = 1.0 * get_select_vector(DF_TEST, POSITIVE_LABELS)



print("TR: ",LABELS_BINARIZED_TRAIN.columns)
print("TS: ",LABELS_BINARIZED_TEST.columns)

TOTAL_TRAIN_EXAMPLES_BY_CLASS = LABELS_BINARIZED_TRAIN.sum(axis=0)
TOTAL_TEST_EXAMPLES_BY_CLASS = LABELS_BINARIZED_TEST.sum(axis=0)
TOTAL_TRAIN_TEST_EXAMPLES_BY_CLASS = TOTAL_TRAIN_EXAMPLES_BY_CLASS + TOTAL_TEST_EXAMPLES_BY_CLASS
TOTAL_TRAIN_TEST_EXAMPLES = LABELS_BINARIZED_TRAIN.shape[0] + LABELS_BINARIZED_TEST.shape[0]

print("TRAIN NUMBER EXAMPLES (BY CLASS):")
print(TOTAL_TRAIN_EXAMPLES_BY_CLASS)
print("TEST NUMBER EXAMPLES (BY CLASS):")
print(TOTAL_TRAIN_TEST_EXAMPLES_BY_CLASS)
print("TOTAL NUMBER EXAMPLES (BY CLASS):")
print(TOTAL_TRAIN_TEST_EXAMPLES_BY_CLASS)
print("PERCENT EXAMPLES (BY CLASS):")
print(TOTAL_TRAIN_TEST_EXAMPLES_BY_CLASS / TOTAL_TRAIN_TEST_EXAMPLES)
########################################################################
      # preprecess the data into required structure
########################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
X_TEST_STANDARDIZED = X_TEST / 255

########## Make a function to read pool #######
def read_pool(pool_path):
    with open(pool_path, "rb") as f:
        df = pickle.load(f)

    split_wavfiles = []
    split_feats = []
    for ii, feat in enumerate(df.features):

        for i in range(0, (len(feat) // 10) * 10, 10):
            if i==0:
                res = feat[i:i+10]
            else:
                res = np.concatenate([res,feat[i:i+10]])

            #split_feats.append(feat[i:i + 10])
            split_wavfiles.append(df.wav_file[ii] + "_start_" + str(i))

        if ii==0:
            super_res = res
        else:
            super_res = np.concatenate([super_res,res])
    #df = pd.DataFrame({"features": split_feats, "wav_files": split_wavfiles})
    #df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
    return super_res.reshape(-1,1280),split_wavfiles
########################################################################
      # create the keras model.
      # Change and play around with model architecture
      # Hyper parameters can be tweaked here
########################################################################

df_pool,wavfiles_pool = read_pool("/home/madlad/Code_Projects/modular_acoustic_detection/diff_class_datasets/field_dataset.pkl")

#features = df_pool.features.values
#labels = df_pool.wav_files.values
print("POOL: ",df_pool.shape)
#print("F SHAPE ",features.shape)
#print("L SHAPE ",labels.shape)

def create_keras_model():
    """
    Creating a Model
    """
    model = Sequential()
    model.add(Conv1D(500, input_shape=(1280, 1), kernel_size=128,
                     strides=128, activation='relu', padding='same'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model



classifier = KerasClassifier(create_keras_model)
########################################################################
    # reshaping the train and test data so as to align with input for model
########################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TEST = X_TEST.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET = LABELS_BINARIZED_TRAIN
CLF2_TEST_TARGET = LABELS_BINARIZED_TEST



########################################################################
      # Implementing & Training the keras model
########################################################################

if config["networkCfgJson"] is None:
    MODEL = create_keras_model()
else:
    json_file = open(config["networkCfgJson"], 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    MODEL = model_from_json(loaded_model_json)
    MODEL.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])

epochs = config["train"]["epochs"]
batch_size = config["train"]["batchSize"]


n_initial = 1000
initial_idx = np.random.choice(range(len(CLF2_TRAIN_TARGET)), size=n_initial, replace=False)
X_initial = CLF2_TRAIN[initial_idx]
y_initial = CLF2_TRAIN_TARGET.iloc[initial_idx].values
X_test = CLF2_TEST
y_test = CLF2_TEST_TARGET.values
print("SHAPE: ",CLF2_TRAIN.shape)

########################## Read Pool #####################################
X_pool = df_pool
X_pool = X_pool.reshape(-1,1280,1)
#X_pool = np.delete(CLF2_TRAIN, initial_idx, axis=0)[:500]
#y_pool = np.delete(CLF2_TRAIN_TARGET.values, initial_idx, axis=0)[:500]





learner = ActiveLearner(
    estimator=classifier,
    X_training=X_initial, y_training=y_initial,
    verbose=1
)


n_queries = 2
query_list = []
model_accuracy = []
for idx in range(n_queries):
    print('Query no. %d' % (idx + 1))
    query_idx, query_instance = learner.query(X_pool, n_instances=3, verbose=0)

    #labels = np.zeros(shape=(100,7))
    features = pd.DataFrame()
    labels = [["label"] for i in range(len(query_idx))]

    features["wavefiles"] = np.array(wavfiles_pool)[query_idx]
    features["labels_name"] = labels
    features.to_csv("active_learner.csv")
    inp = input("Enter any character after you finish labelling the csv")
    labelled_csv = pd.read_csv("active_learner.csv",index_col=0)
    labs = pd.DataFrame()

    for key in pos_sounds.keys():
        FULL_NAME = key
        POSITIVE_LABELS = pos_sounds[key]
        labs[FULL_NAME] = 1.0 * get_select_vector(labelled_csv, POSITIVE_LABELS)

    y_pool = labs.values


    learner.teach(
        X=X_pool[query_idx], y=y_pool, only_new=True,
        verbose=1
    )
    model_accuracy.append(learner.score(X_test,y_test))
    query_list.append(idx)
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)

print("Acc: ",model_accuracy)
print("Query: ",query_list)
plt.plot(query_list,model_accuracy)
plt.show()
