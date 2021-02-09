"""
Training a Multilabel Model
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from tensorflow.compat.v1.keras.models import Sequential, model_from_json
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.optimizers import Adam
from youtube_audioset import get_recursive_sound_names
import os
from glob import glob
import json
from colorama import Fore, Style
import argparse
from tensorflow.keras.callbacks import EarlyStopping
from keras_balanced_batch_generator import make_generator

#########################################################
# Description and Help
#########################################################
DESCRIPTION = "Reads config file from user and trains multilabel model accordingly."
HELP = "Input config filepath for the required model to be trained."
#########################################################
# Parse the arguments
#########################################################
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("-cfg_json", action="store", help=HELP, required=True)
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

assert (config["ontology"]["useYoutubeAudioSet"])
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
    comprising_pos_labels = get_recursive_sound_names(label_dicts["positiveLabels"], "./", ontologyExtFiles)
    pos_sounds[lab_name] = comprising_pos_labels
    if label_dicts["negativeLabels"] is not None:
        neg_lab_name = label_dicts["aggregateNegativeLabelName"]
        comprising_neg_labels = get_recursive_sound_names(label_dicts["negativeLabels"], "./", ontologyExtFiles)
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
    test_size = int(validation_split * dataframe.shape[0])

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

    test_subsample = int(subsample * validation_split)
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
            print(Fore.RED, "No file matches pattern criteria:", pattern_path, "Excluded Paths:", exclude_paths,
                  Style.RESET_ALL)
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
        assert ("patternPath" not in list(input_file_dict.keys()))
        assert ("path" in list(input_file_dict.keys()))

        print("Importing", input_file_dict["path"], "...")

        with open(input_file_dict["path"], 'rb') as file_obj:
            # Load the file

            df = pickle.load(file_obj)

            # Filtering the sounds that are exactly 10 seconds
            # Examples should be exactly 10 seconds. Anything else
            # is not a valid input to the model
            df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]

            train_file_examples_df, test_file_examples_df = \
                split_and_subsample_dataframe(dataframe=df,
                                              validation_split=validation_split,
                                              subsample=input_file_dict["subsample"])

            # append to overall list of examples
            list_of_train_dataframes.append(train_file_examples_df)
            list_of_test_dataframes.append(test_file_examples_df)

    train_df = pd.concat(list_of_train_dataframes, ignore_index=True)
    test_df = pd.concat(list_of_test_dataframes, ignore_index=True)

    print("Import done.")

    return train_df, test_df


DF_TRAIN, DF_TEST = import_dataframes(dataframe_file_list=config["train"]["inputDataFrames"],
                                      positive_label_filter_arr=pos_sounds,
                                      negative_label_filter_arr=neg_sounds,
                                      validation_split=config["train"]["validationSplit"])

LABELS_BINARIZED_TRAIN = pd.DataFrame()
LABELS_BINARIZED_TEST = pd.DataFrame()
for key in pos_sounds.keys():
    FULL_NAME = key
    POSITIVE_LABELS = pos_sounds[key]
    LABELS_BINARIZED_TRAIN[FULL_NAME] = 1.0 * get_select_vector(DF_TRAIN, POSITIVE_LABELS)

    LABELS_BINARIZED_TEST[FULL_NAME] = 1.0 * get_select_vector(DF_TEST, POSITIVE_LABELS)

print("TR: ", LABELS_BINARIZED_TRAIN.columns)
print("TS: ", LABELS_BINARIZED_TEST.columns)

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
# preprocess the data into required structure
########################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())


########################################################################
# create the keras model.
# Change and play around with model architecture
# Hyper parameters can be tweaked here
########################################################################
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

callback = EarlyStopping(
    monitor="val_loss",
    verbose=1,
    mode="auto"
)

training_generator = make_generator(
    CLF2_TRAIN, CLF2_TRAIN_TARGET.values, batch_size=batch_size, categorical=True)

'''MODEL_TRAINING = MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                           epochs=epochs, batch_size=batch_size, verbose=1,
                           callbacks = [callback],
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET))'''

steps_per_epoch = len(CLF2_TRAIN) // batch_size
MODEL_TRAINING = MODEL.fit(training_generator, shuffle=True, epochs=2, steps_per_epoch=steps_per_epoch,
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET.values), verbose=1)

########################################################################
# Predict on train and test data
# Changing decision threshold can be done here
########################################################################
CLF2_TRAIN_PREDICTION = MODEL.predict(CLF2_TRAIN).round()
CLF2_TRAIN_PREDICTION_PROB = MODEL.predict(CLF2_TRAIN)
CLF2_TEST_PREDICTION = MODEL.predict(CLF2_TEST).round()
CLF2_TEST_PREDICTION_PROB = MODEL.predict(CLF2_TEST)

########################################################################
# To get the Misclassified examples
########################################################################
DF_TEST['actual_labels'] = np.split(LABELS_BINARIZED_TEST.values, DF_TEST.shape[0])
DF_TEST['predicted_labels'] = np.split(CLF2_TEST_PREDICTION, DF_TEST.shape[0])
DF_TEST['predicted_prob'] = np.split(CLF2_TEST_PREDICTION_PROB, DF_TEST.shape[0])

MISCLASSIFED_ARRAY = CLF2_TEST_PREDICTION != CLF2_TEST_TARGET
MISCLASSIFIED_EXAMPLES = np.any(MISCLASSIFED_ARRAY, axis=1)

########################################################################
# print misclassified number of examples
########################################################################
print('Misclassified number of examples :', DF_TEST[MISCLASSIFIED_EXAMPLES].shape[0])

########################################################################
# Print confusion matrix and classification_report
########################################################################
print('        Confusion Matrix          ')
print('============================================')
for i in range(CLF2_TEST_TARGET.shape[1]):
    print("Confusion matrix for ", CLF2_TEST_TARGET.columns[i])
    a = CLF2_TEST_TARGET[CLF2_TEST_TARGET.columns[i]].values
    b = CLF2_TEST_PREDICTION[:, i]
    RESULT_ = confusion_matrix(a, b)
    print(RESULT_)
print('        Classification Report      ')
print('============================================')
CL_REPORT = classification_report(CLF2_TEST_TARGET.values.argmax(axis=1),
                                  CLF2_TEST_PREDICTION.argmax(axis=1))
print(CL_REPORT)

########################################################################
# calculate accuracy and hamming loss
########################################################################
ACCURACY = accuracy_score(CLF2_TEST_TARGET.values.argmax(axis=1),
                          CLF2_TEST_PREDICTION.argmax(axis=1))
HL = hamming_loss(CLF2_TEST_TARGET.values.argmax(axis=1), CLF2_TEST_PREDICTION.argmax(axis=1))
print('Hamming Loss :', HL)
print('Accuracy :', ACCURACY)

########################################################################
# Save the model weights
# Change the name if you are tweaking hyper parameters
########################################################################
MODEL.save_weights(config["train"]["outputWeightFile"])
