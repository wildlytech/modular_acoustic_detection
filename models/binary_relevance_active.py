"""
Traning a Binary Relevance Model
"""
#Import the necessary functions and libraries
import argparse
from collections import Counter
import random

from colorama import Fore, Style
from glob import glob
import json
from tensorflow.compat.v1.keras.models import model_from_json, Sequential
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.optimizers import Adam
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt
from modAL.batch import uncertainty_batch_sampling
from modAL.uncertainty import uncertainty_sampling
import sys
import datetime
import tensorflow as tf
from youtube_audioset import get_recursive_sound_names
from sklearn.metrics import accuracy_score,precision_score,recall_score
from tensorflow.compat.v1.keras.models import Model
#############################################################################
                # Description and help
#############################################################################

DESCRIPTION = "Reads the configuration file to train a particular \
               label and outputs the model"

#############################################################################
          # Parse the input arguments given from command line
#############################################################################

ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()
REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
REQUIRED_NAMED.add_argument('-model_cfg_json', '--model_cfg_json',
                            help='Input json configuration file for label',
                            required=True)
OPTIONAL_NAMED.add_argument('-output_weight_file', '--output_weight_file', help='Output weight file name')
ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
PARSED_ARGS = ARGUMENT_PARSER.parse_args()

#############################################################################
          # Parse the arguments
#############################################################################
### SET SEEDS
np.random.seed(10)
random.seed(10)
tf.compat.v1.set_random_seed(10)


with open(PARSED_ARGS.model_cfg_json) as json_file_obj:
  CONFIG_DATA = json.load(json_file_obj)

FULL_NAME = CONFIG_DATA["aggregatePositiveLabelName"] + 'vs' + CONFIG_DATA["aggregateNegativeLabelName"]

if PARSED_ARGS.output_weight_file is None:
  # Set default output weight file name
  OUTPUT_WEIGHT_FILE = CONFIG_DATA["train"]["outputWeightFile"]
else:
  # Use argument weight file name
  OUTPUT_WEIGHT_FILE = PARSED_ARGS.output_weight_file

# create directory to the folder
pathToFileDirectory = "/".join(OUTPUT_WEIGHT_FILE.split('/')[:-1]) + '/'
if not os.path.exists(pathToFileDirectory):
  os.makedirs(pathToFileDirectory)

#############################################################################
          # Get all sound names
#############################################################################

# Model training only supports using audio set as main ontology
assert(CONFIG_DATA["ontology"]["useYoutubeAudioSet"])

# List of paths to json files that will be used to extend
# existing youtube ontology
ontologyExtFiles = CONFIG_DATA["ontology"]["extension"]

# If single file or null, then convert to list
if ontologyExtFiles is None:
  ontologyExtFiles = []
elif type(ontologyExtFiles) != list:
  ontologyExtFiles = [ontologyExtFiles]

# Grab all the positive labels
POSITIVE_LABELS = get_recursive_sound_names(CONFIG_DATA["positiveLabels"], "./", ontologyExtFiles)

# If negative labels were provided, then collect them
# Otherwise, assume all examples that are not positive are negative
if CONFIG_DATA["negativeLabels"] is None:
  NEGATIVE_LABELS = None
else:
  # Grab all the negative labels
  NEGATIVE_LABELS = get_recursive_sound_names(CONFIG_DATA["negativeLabels"], "./", ontologyExtFiles)
  # Make sure there is no overlap between negative and positive labels
  NEGATIVE_LABELS = NEGATIVE_LABELS.difference(POSITIVE_LABELS)

#############################################################################
          # Importing dataframes from the function
#############################################################################


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
    assert(False)
  else:
    train_df = subsample_dataframe(train_df, train_subsample)

  if (test_df.shape[0] == 0) and (test_subsample > 0):
    print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)
    assert(False)
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
  pattern_file_dicts =  [x for x in dataframe_file_list if "patternPath" in list(x.keys())]

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
      print(Fore.RED, "No file matches pattern criteria:", pattern_path, "Excluded Paths:", exclude_paths, Style.RESET_ALL)
      assert(False)

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

      # Only use examples that have a label in label filter array
      positive_example_select_vector = get_select_vector(df, positive_label_filter_arr)
      positive_examples_df = df.loc[positive_example_select_vector]

      # This ensures there no overlap between positive and negative examples
      negative_example_select_vector = ~positive_example_select_vector
      if negative_label_filter_arr is not None:
        # Exclude even further examples that don't fall into the negative label filter
        negative_example_select_vector &= get_select_vector(df, negative_label_filter_arr)
      negative_examples_df = df.loc[negative_example_select_vector]

      # No longer need df after this point
      del df

      train_positive_examples_df, test_positive_examples_df = \
                split_and_subsample_dataframe(dataframe=positive_examples_df,
                                              validation_split=validation_split,
                                              subsample=input_file_dict["positiveSubsample"])
      del positive_examples_df

      train_negative_examples_df, test_negative_examples_df = \
                split_and_subsample_dataframe(dataframe=negative_examples_df,
                                              validation_split=validation_split,
                                              subsample=input_file_dict["negativeSubsample"])
      del negative_examples_df

      # append to overall list of examples
      list_of_train_dataframes += [train_positive_examples_df, train_negative_examples_df]
      list_of_test_dataframes += [test_positive_examples_df, test_negative_examples_df]

  train_df = pd.concat(list_of_train_dataframes, ignore_index=True)
  test_df = pd.concat(list_of_test_dataframes, ignore_index=True)

  print("Import done.")

  return train_df, test_df

DF_TRAIN, _ = \
    import_dataframes(dataframe_file_list=CONFIG_DATA["train"]["inputDataFrames"],
                      positive_label_filter_arr=POSITIVE_LABELS,
                      negative_label_filter_arr=NEGATIVE_LABELS,
                      validation_split=CONFIG_DATA["train"]["validationSplit"])

with open("diff_class_datasets/Datasets/Nandi.pkl","rb") as f:
    DF_TEST = pickle.load(f)



#DF_TRAIN = DF_TRAIN.sample(frac=0.1)
pool_df = DF_TEST.sample(frac=0.2)
#DF_TRAIN = DF_TRAIN.drop(pool_df.index)
DF_TEST = DF_TEST.drop(pool_df.index)
X_pool = np.array(pool_df.features.apply(lambda x: x.flatten()).tolist())
drop_indices_pool = []
new_x_pool = []
for ii,i in enumerate(X_pool):

    if len(i)!=1280:

        drop_indices_pool.append(ii)
    else:
        new_x_pool.append(i)
#X_pool = np.delete(X_pool,drop_indices_pool)
X_pool = np.array(new_x_pool)
X_pool_STANDARDIZED = X_pool / 255
CLF2_pool = X_pool.reshape((-1, 1280, 1))
LABELS_BINARIZED_pool = pd.DataFrame()
LABELS_BINARIZED_pool[FULL_NAME] = 1.0 * get_select_vector(pool_df, POSITIVE_LABELS)
CLF2_pool_target = LABELS_BINARIZED_pool.values
CLF2_pool_target = np.delete(CLF2_pool_target,drop_indices_pool)
#############################################################################
        # Turn the target labels into one binarized vector
#############################################################################
LABELS_BINARIZED_TRAIN = pd.DataFrame()
LABELS_BINARIZED_TRAIN[FULL_NAME] = 1.0 * get_select_vector(DF_TRAIN, POSITIVE_LABELS)

LABELS_BINARIZED_TEST = pd.DataFrame()
LABELS_BINARIZED_TEST[FULL_NAME] = 1.0 * get_select_vector(DF_TEST, POSITIVE_LABELS)

#############################################################################
        # print out the number and percentage of each class examples
#############################################################################

TOTAL_TRAIN_TEST_EXAMPLES = LABELS_BINARIZED_TRAIN.shape[0] + LABELS_BINARIZED_TEST.shape[0]
TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES = (LABELS_BINARIZED_TRAIN[FULL_NAME] == 1).sum() + \
                                     (LABELS_BINARIZED_TEST[FULL_NAME] == 1).sum()
TOTAL_TRAIN_TEST_NEGATIVE_EXAMPLES = (LABELS_BINARIZED_TRAIN[FULL_NAME] == 0).sum() + \
                                     (LABELS_BINARIZED_TEST[FULL_NAME] == 0).sum()

print("NUMBER EXAMPLES (TOTAL/POSITIVE/NEGATIVE):", \
      TOTAL_TRAIN_TEST_EXAMPLES, "/", \
      TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES, "/", \
      TOTAL_TRAIN_TEST_NEGATIVE_EXAMPLES)

print("PERCENT POSITIVE EXAMPLES:", "{0:.2f}%".format(100.0*TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES/TOTAL_TRAIN_TEST_EXAMPLES))

#############################################################################
        # preprocess the data into required structure
#############################################################################



X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = DF_TEST.features.apply(lambda x: x.flatten()).tolist()

print(len(LABELS_BINARIZED_TEST))
drop_indices = []
for ii,i in enumerate(X_TEST):

    if len(i)!=1280:
        X_TEST.remove(i)
        drop_indices.append(ii)

X_TEST = np.array(X_TEST)

X_TEST_STANDARDIZED = X_TEST / 255

print(LABELS_BINARIZED_TEST.value_counts())

#############################################################################
        # create the keras model. It is a maxpool version BR model
#############################################################################
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

    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    #model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/WildAnimals_BR_Model/4_Layer_Variants/binary_relevance_WildAnimals_realised_weights_puresounds_TM_TD_TH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    #print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model


#############################################################################
   # reshaping the train and test data so as to align with input for model
#############################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TEST = X_TEST.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET = LABELS_BINARIZED_TRAIN.values
CLF2_TEST_TARGET = LABELS_BINARIZED_TEST.values

CLF2_TEST_TARGET = np.delete(CLF2_TEST_TARGET,drop_indices)
#############################################################################
   # assign class weights to ensure balanced datasets during training
#############################################################################

TRAIN_TARGET_POSITIVE_PERCENTAGE = CLF2_TRAIN_TARGET.mean()

print("TRAIN NUMBER EXAMPLES (TOTAL/POSITIVE/NEGATIVE):", \
      CLF2_TRAIN_TARGET.shape[0], "/", \
      (CLF2_TRAIN_TARGET == 1).sum(), "/", \
      (CLF2_TRAIN_TARGET == 0).sum())
print("TRAIN PERCENT POSITIVE EXAMPLES:", "{0:.2f}%".format(100*TRAIN_TARGET_POSITIVE_PERCENTAGE))

if TRAIN_TARGET_POSITIVE_PERCENTAGE > 0.5:
  CLASS_WEIGHT_0 = TRAIN_TARGET_POSITIVE_PERCENTAGE / (1-TRAIN_TARGET_POSITIVE_PERCENTAGE)
  CLASS_WEIGHT_1 = 1
else:
  CLASS_WEIGHT_0 = 1
  CLASS_WEIGHT_1 = (1-TRAIN_TARGET_POSITIVE_PERCENTAGE) / TRAIN_TARGET_POSITIVE_PERCENTAGE

#############################################################################
      # Implementing using the keras usual training techinque
#############################################################################

json_file = open(CONFIG_DATA["networkCfgJson"], 'r')
loaded_model_json = json_file.read()
mismatch_model = model_from_json(loaded_model_json)



def train_model(CONFIG_DATA):
    if CONFIG_DATA["networkCfgJson"] is None:
      MODEL = create_keras_model()
    else:
      # load json and create model
      json_file = open(CONFIG_DATA["networkCfgJson"], 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      MODEL = model_from_json(loaded_model_json)
      #mismatch_model = model_from_json(loaded_model_json)

      #mismatch_model.load_weights(CONFIG_DATA["train"]["outputWeightFile"])
      MODEL.load_weights(CONFIG_DATA["train"]["outputWeightFile"])
      MODEL.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5,epsilon=1e-8),
                    metrics=['accuracy'])

    #CONFIG_DATA["train"]["epochs"]
    '''MODEL_TRAINING = MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                               epochs=1,
                               batch_size=CONFIG_DATA["train"]["batchSize"],
                               class_weight={0: CLASS_WEIGHT_0, 1: CLASS_WEIGHT_1},
                               verbose=1,
                               validation_data=(CLF2_TEST, CLF2_TEST_TARGET))'''
    return MODEL

#MODEL,MODEL_TRAINING = train_model(CONFIG_DATA,CLF2_TRAIN,CLF2_TRAIN_TARGET,CLF2_TEST,CLF2_TEST_TARGET,CLASS_WEIGHT_0,CLASS_WEIGHT_1)
#### MISMATCH MODEL ####
MODEL = train_model(CONFIG_DATA)
mismatch_model = Model(MODEL.input,MODEL.layers[-3].output)
#mismatch_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-7, epsilon=1e-8),
#                metrics=['accuracy'])

'''MODEL.save("binary_relevance.h5")
plt.plot(MODEL_TRAINING.history['accuracy'])
plt.plot(MODEL_TRAINING.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("train_acc_vs_val_acc.png")
plt.show()'''

def read_pool(pool_path):
    with open(pool_path, "rb") as f:
        df = pickle.load(f)
    #df = pd.read_csv(pool_path)
    split_wavfiles = []
    split_labels = []
    for ii, feat in enumerate(df.features):

        for i in range(0, (len(feat) // 10) * 10, 10):
            if i==0:
                res = feat[i:i+10]
            else:
                res = np.concatenate([res,feat[i:i+10]])

            #split_feats.append(feat[i:i + 10])
            split_wavfiles.append(df.wav_file[ii] + "_start_" + str(i))
            split_labels.append(df.label_name[ii])
        if ii==0:
            super_res = res
        else:
            super_res = np.concatenate([super_res,res])
    #df = pd.DataFrame({"features": split_feats, "wav_files": split_wavfiles})
    #df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
    return super_res.reshape(-1,1280),split_wavfiles,split_labels



print("POOL LABELS: ",LABELS_BINARIZED_pool.value_counts())
#df_pool,wavfiles_pool,labels_pool = read_pool("active_learning_test.pkl")

classifier = KerasClassifier(create_keras_model)
X_initial = CLF2_TRAIN
y_initial = CLF2_TRAIN_TARGET
X_test = CLF2_TEST
y_test = CLF2_TEST_TARGET


'''X_pool = df_pool
X_pool = X_pool.reshape(-1,1280,1)
t_indices = np.random.randint(len(X_pool),size=30)
pool_test_X = X_pool[t_indices,:]

pool_test_Y = [labels_pool[index] for index in t_indices]
temp_df = pd.DataFrame()
temp_df["labels_name"] = pool_test_Y
pool_test_Y = 1.0*get_select_vector(temp_df,POSITIVE_LABELS)
pool_test_Y = pool_test_Y.values.reshape(-1,1)
X_test = np.concatenate([X_test,pool_test_X],axis=0)
y_test = np.concatenate([y_test,pool_test_Y],axis=0)


X_pool = np.delete(X_pool,t_indices,axis=0)
labels_pool = [labels_pool[index] for index in range(len(labels_pool)) if index not in t_indices]'''



'''learner = ActiveLearner(
    estimator=classifier,
    query_strategy=uncertainty_sampling,
    X_training=X_initial, y_training=y_initial,

    verbose=1,**{"epochs":1}
)'''


def convert2list(x):
    list_vals =[]
    for val in x:
        val =val.lstrip("[")
        val = val.rstrip("]")
        if "," in val:
            ls = val.split(",")
            list_vals.append(ls)
        else:

            list_vals.append([val.strip("'")])
    return list_vals

n_queries = 20
query_list = []
model_accuracy = []


#preds = learner.predict(X_test)
#true = y_test
#print("INIT ACC: ",accuracy_score(true,preds))

labels_pool_arr = LABELS_BINARIZED_pool.values


def query_KNN(labelled_truth,labelled_truth_labs,pool_df,pool_labs,n_instances):
    '''
    labelled_truth: Embeddings of the labelled audio
    preds: Prediction probabilities of each second
    n_instances: Number of instances
    '''

    random.seed(42)


    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(labelled_truth[:,:,0],labelled_truth_labs[:,0])
    pool_zs = np.where(pool_labs==0)[0]
    pool_ones = np.where(pool_labs==1)[0]
    #knn_dist,knn_ind = knn.kneighbors(pool_df[:,:,0])
    print("Getting NN's...")
    print("DF: ",len(pool_df),"Labels: ",len(pool_labs))
    z_knn_dist,z_knn_ind = knn.kneighbors(pool_df[pool_zs,:,0])
    o_knn_dist,o_knn_ind = knn.kneighbors(pool_df[pool_ones,:,0])

    print("Getting Indices...")
    z_knn_dist = np.ravel(z_knn_dist)
    o_knn_dist = np.ravel(o_knn_dist)

    z_indices = (-z_knn_dist).argsort()
    o_indices = (-o_knn_dist).argsort()
    #labels = pool_labs.values[indices[:n_instances]]
    final_inds = np.concatenate([z_indices[:n_instances//2],o_indices[:n_instances//2]],axis=0)
    #embeddings = pool_df[indices[:n_instances]]
    #return indices[:n_instances]

    print("FI: ",final_inds.shape)
    return final_inds

def get_ground_probs(mismatch_model,train_df):
    ground_probs = mismatch_model.predict(train_df)
    ground_feats = train_df.reshape(-1, 128, 1)
    return ground_feats,ground_probs.reshape(-1)

ground_feats,ground_probs = get_ground_probs(mismatch_model,CLF2_TRAIN)

def active_mismatch(mismatch_model,pool_df,ground_feats,ground_probs,n_instances=10):
    print("Predicting on pool: ")
    model_probs = mismatch_model.predict(pool_df)
    pool_feats = pool_df.reshape(-1,128,1)
    print("MP: ",model_probs.shape)
    model_probs = model_probs.reshape(-1)

    print("Predicting on KNN: ")
    knn = KNeighborsRegressor(n_neighbors=2,n_jobs=2)
    knn.fit(ground_feats[:,:,0],ground_probs.reshape(-1))
    knn_probs = knn.predict(pool_feats[:,:,0])

    prob_diff = abs(model_probs-knn_probs)
    indices = (-prob_diff).argsort()
    indices = indices//10
    return indices[:n_instances]

def specificity(true,preds):
    req  = preds[preds==true]

    num_tn = len(req[req==0])
    num_total_n = len(true[true==0])
    return num_tn/num_total_n






def query_entropy(model,pool_df,n_instances):
    pool_preds = model.predict(pool_df)
    entropy = -pool_preds*np.log(pool_preds)
    entropy = np.ravel(entropy)
    indices = (-entropy).argsort()
    return_probs = pool_preds[indices[:n_instances]]
    return indices[:n_instances],return_probs


def random_sampling(len_df,n_instances):
    random_indices = np.array(random.sample(range(len_df),n_instances))
    return random_indices


def cosine_distance(m1,m2):

    sim =  np.dot(m1.transpose(),m2)/(np.linalg.norm(m1)*np.linalg.norm(m2))
    return 1-sim

def farthest_first(cur_pool,total_pool):
    farthest_indices = []
    for X in cur_pool:


        max_dist = 0
        max_index=0
        for ii,Y in enumerate(total_pool):

            cos_dist = cosine_distance(X,Y)


            if cos_dist>max_dist and ii not in farthest_indices:
                max_dist = cos_dist
                max_index = ii
        farthest_indices.append(max_index)
    return farthest_indices


def active_learning_loop(CLF2_pool,LABELS_BINARIZED_pool,CLF2_TRAIN,CLF2_TRAIN_TARGET,n_queries,strategy=None):
    labels_pool_arr = LABELS_BINARIZED_pool
    model_accuracy = []
    query_list = []
    prec = []
    rec = []
    tnr = []
    zeroes = np.where(CLF2_TRAIN_TARGET==0)[0]
    ones = np.where(CLF2_TRAIN_TARGET==1)[0]
    MODEL = train_model(CONFIG_DATA)

    preds = MODEL.predict(X_test)
    query_list.append(0)
    model_accuracy.append(accuracy_score(y_test,np.round(preds)))
    prec.append(precision_score(y_test,np.round(preds)))
    rec.append(recall_score(y_test,np.round(preds)))

    tnr.append(specificity(y_test,np.round(preds).reshape(-1)))
    mismatch_model = Model(MODEL.input, MODEL.layers[-3].output)
    print("CONFUSION: ",confusion_matrix(y_test,np.round(preds)))
    def equality(l1,l2):

        for el1,el2 in zip(l1,l2):

            if np.array_equal(el1,el2):
                print("Equal weights found")

    prev_ls = mismatch_model.get_weights()
    train_acc = []
    queries = []
    prob_t_plot = []
    for idx in range(n_queries):

        print('Query no. %d' % (idx + 1))

        #query_idx, query_instance = learner.query(CLF2_pool, n_instances=10, verbose=0)
        #preds = MODEL.predict(CLF2_pool)
        if strategy=="active":
            #man_idx = query_KNN(CLF2_TRAIN,CLF2_TRAIN_TARGET,CLF2_pool,labels_pool_arr,10)
            man_idx,ret_probs = query_entropy(MODEL,CLF2_pool,10)
            prob_t_plot.append(ret_probs)
            #man_idx  = active_mismatch(mismatch_model,CLF2_pool,ground_feats,ground_probs,n_instances=10)
        elif strategy=="random":
            man_idx = random_sampling(len(CLF2_pool),10)
        #print("M: ",man_idx)

        #labels = np.zeros(shape=(100,7))
        #features = pd.DataFrame()
        #labels = [["label"] for i in range(len(query_idx))]

        #features["wavefiles"] = np.array(wavfiles_pool)[query_idx]
        #features["labels_name"] = np.array(labels_pool)[query_idx].tolist()

        #features.to_csv("active_learner.csv")

        #inp = input("Enter any character after you finish labelling the csv")
        #labelled_csv = pd.read_csv("active_learner.csv",index_col=0)
        #label_list = convert2list(labelled_csv.labels_name)
        #labelled_csv = features
        #labs = pd.DataFrame()
        #labelled_csv["labels_name"] = label_list
        #labs["labels_name"] = 1.0*get_select_vector(labelled_csv,POSITIVE_LABELS)

        #y_pool = labs.values
        #y_pool = labels_pool_arr[query_idx]


        #print("YPOOL: ",np.argmax(y_pool,axis=1))
        '''learner.teach(
            X=CLF2_pool[query_idx], y=y_pool, only_new=True,
            verbose=1,**{"epochs": 10}
        )'''
        #farthest_indices = farthest_first(CLF2_pool[man_idx],CLF2_pool)
        #man_idx = list(man_idx)
        #man_idx.extend(farthest_indices)
        print("L: ",len(labels_pool_arr))
        print("IDX: ",man_idx)
        y_pool = labels_pool_arr[man_idx]
        unique, counts = np.unique(y_pool, return_counts=True)
        value_counts = dict(zip(unique, counts))
        print("Counter Query: ", dict(zip(unique, counts)))


        print("CLASS BALANCING")
        if 0.0 not in value_counts.keys():
            value_counts[0.0]=0
        if 1.0 not in value_counts.keys():
            value_counts[1.0] = 0
        extra_zs = 10-value_counts[0]
        extra_ons = 10-value_counts[1]



        choice_z = np.random.choice(len(zeroes),extra_zs)
        choice_o = np.random.choice(len(ones),extra_ons)

        z_index = zeroes[choice_z]

        o_index = ones[choice_o]



        final_query_x = np.concatenate([CLF2_pool[man_idx],CLF2_TRAIN[z_index],CLF2_TRAIN[o_index]],axis=0)
        final_query_y = np.concatenate([y_pool,CLF2_TRAIN_TARGET[z_index].reshape(-1),CLF2_TRAIN_TARGET[o_index].reshape(-1)],axis=0)


        unique, counts = np.unique(final_query_y, return_counts=True)

        print("Final Counter Query: ", dict(zip(unique, counts)))

        #MODEL.train_on_batch(CLF2_pool[man_idx],y=y_pool)
        ret = MODEL.train_on_batch(final_query_x,final_query_y)

        train_acc.append(ret[0])
        #if strategy=="active":
        #    queries.append((final_query_x,final_query_y))

        #ls = mismatch_model.get_weights()
        #equality(ls, prev_ls)
        '''json_file = open(CONFIG_DATA["networkCfgJson"], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        MODEL = model_from_json(loaded_model_json)
        MODEL.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                      metrics=['accuracy'])
        print("Re training model: ")
        X_initial = np.concatenate([X_initial,X_pool[query_idx]])
        y_initial = np.concatenate([y_initial,y_pool])
        MODEL_TRAINING = MODEL.fit(X_initial, y_initial,
                                   epochs=CONFIG_DATA["train"]["epochs"],
                                   batch_size=CONFIG_DATA["train"]["batchSize"],
                                   class_weight={0: CLASS_WEIGHT_0, 1: CLASS_WEIGHT_1},
                                   verbose=1,
                                   validation_data=(CLF2_TEST, CLF2_TEST_TARGET))'''

        #model_accuracy.append(learner.score(X_test,y_test))
        #print("PRED: ",learner.predict(X_test).reshape(1,-1).count(0))
        #print("Y: ",y_test.reshape(1,-1))

        preds = np.round(MODEL.predict(X_test))
        #preds = learner.predict(X_test)

        #true = y_test
        #preds = np.round(learner.predict_proba(X_test))

        true = y_test
        print("T: ",true)
        print("P: ",preds)
        true = true.reshape(1,-1)[0]
        assert len(true)==len(preds)
        acc = accuracy_score(true,preds)
        model_accuracy.append(acc)
        #print("BOOL ACCURACY: ",np.sum(true==preds)/len(true))
        print("SCIKIT ACCURACY: ",accuracy_score(true,preds))
        query_list.append(idx+1)
        prec.append(precision_score(true,preds))
        rec.append(recall_score(true,preds))
        tnr.append(specificity(true,preds.reshape(-1)))
        # remove queried instance from pool
        #CLF2_pool = np.delete(CLF2_pool, query_idx, axis=0)
        #labels_pool_arr = np.delete(labels_pool_arr,query_idx,axis=0)
        CLF2_pool = np.delete(CLF2_pool, man_idx, axis=0)
        labels_pool_arr = np.delete(labels_pool_arr, man_idx, axis=0)
        #prev_ls = ls
    if strategy=="active":
        with open("queried_ex.pkl", "wb") as f:
            pickle.dump(queries,f)
    plt.plot(query_list[1:],train_acc)
    plt.xlabel("Query Index")
    plt.ylabel("Train Loss")
    #plt.savefig("Accuracy_vs_query.png")
    plt.show()
    return query_list,model_accuracy,prec,rec,tnr,prob_t_plot

q_list_active,acc_active,prec_active,rec_active,tnr_active,prob_t_plot = active_learning_loop(CLF2_pool,CLF2_pool_target,CLF2_TRAIN,CLF2_TRAIN_TARGET,20,"active")
#q_list_rand,acc_rand,prec_rand,rec_rand,tnr_rand,prob_t_plot = active_learning_loop(CLF2_pool,CLF2_pool_target,CLF2_TRAIN,CLF2_TRAIN_TARGET,20,"random")

plt.plot(np.ravel(np.array(prob_t_plot)))
plt.show()

plt.plot(q_list_active,acc_active,"blue",label="active_learning")
plt.plot(q_list_rand,acc_rand,"red",label="random_sampling")
plt.xlabel("Query Index")
plt.ylabel("Accuracy")
plt.legend(['active_learning', 'random_sampling'], loc='upper left')
plt.savefig("acc_vs_query.png")
plt.show()



plt.plot(q_list_active,prec_active,"blue",label="active_learning")
plt.plot(q_list_rand,prec_rand,"red",label="random_sampling")
plt.xlabel("Query Index")
plt.ylabel("Precision")
plt.legend(['active_learning', 'random_sampling'], loc='upper left')
plt.savefig("precision_vs_query.png")
plt.show()

plt.plot(q_list_active,rec_active,"blue",label="active_learning")
plt.plot(q_list_rand,rec_rand,"red",label="random_sampling")
plt.xlabel("Query Index")
plt.ylabel("Recall")
plt.legend(['active_learning', 'random_sampling'], loc='upper left')
plt.savefig("Recall_vs_query.png")
plt.show()


plt.plot(q_list_active,tnr_active,"blue",label="active_learning")
plt.plot(q_list_rand,tnr_rand,"red",label="random_sampling")
plt.xlabel("Query Index")
plt.ylabel("TNR")
plt.legend(['active_learning', 'random_sampling'], loc='upper left')
plt.savefig("TNR_vs_query.png")
plt.show()


#############################################################################
      # Predict on train and test data
#############################################################################


'''
CLF2_TRAIN_PREDICTION = MODEL.predict(CLF2_TRAIN).round()
CLF2_TRAIN_PREDICTION_PROB = MODEL.predict(CLF2_TRAIN)
CLF2_TEST_PREDICTION = MODEL.predict(CLF2_TEST).round()
CLF2_TEST_PREDICTION_PROB = MODEL.predict(CLF2_TEST)



#############################################################################
      # To get the Misclassified examples
#############################################################################
# DF_TEST['actual_labels'] = np.split(LABELS_BINARIZED_TEST.values, DF_TEST.shape[0])
# DF_TEST['predicted_labels'] = np.split(CLF2_TEST_PREDICTION, DF_TEST.shape[0])
# DF_TEST['predicted_prob'] = np.split(CLF2_TEST_PREDICTION_PROB, DF_TEST.shape[0])
MISCLASSIFED_ARRAY = CLF2_TEST_PREDICTION != CLF2_TEST_TARGET



#############################################################################
        # print misclassified number of examples
#############################################################################
print('Misclassified number of examples :', MISCLASSIFED_ARRAY.sum())



#############################################################################
        # Print confusion matrix and classification_report
#############################################################################
print(CLF2_TEST_TARGET.shape)
print('        Confusion Matrix          ')
print('============================================')
RESULT = confusion_matrix(CLF2_TEST_TARGET,
                          CLF2_TEST_PREDICTION)
print(RESULT)



#############################################################################
        # print classification report
#############################################################################
print('                 Classification Report      ')
print('============================================')
CL_REPORT = classification_report(CLF2_TEST_TARGET,
                                  CLF2_TEST_PREDICTION)
print(CL_REPORT)


#############################################################################
        # calculate accuracy and hamming loss
#############################################################################
ACCURACY = accuracy_score(CLF2_TEST_TARGET,
                          CLF2_TEST_PREDICTION)
HL = hamming_loss(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION)
print('Hamming Loss :', HL)
print('Accuracy :', ACCURACY)



#############################################################################
        # save model weights. Change as per the model type
#############################################################################
MODEL.save_weights(OUTPUT_WEIGHT_FILE)'''

