"""
Traning a Binary Relevance Model
"""
#Import the necessary functions and libraries
import argparse
import json
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
import sys

sys.path.insert(0, "../")
from youtube_audioset import get_recursive_sound_names

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

with open(PARSED_ARGS.model_cfg_json) as json_file_obj:
  CONFIG_DATA = json.load(json_file_obj)

PATH_TO_DIRECTORY_OF_CONFIG = '/'.join(PARSED_ARGS.model_cfg_json.split('/')[:-1]) + '/'

FULL_NAME = CONFIG_DATA["aggregatePositiveLabelName"] + 'vs' + CONFIG_DATA["aggregateNegativeLabelName"]

if PARSED_ARGS.output_weight_file is None:
  # Set default output weight file name
  OUTPUT_WEIGHT_FILE = CONFIG_DATA["train"]["outputWeightFile"]
else:
  # Use argument weight file name
  OUTPUT_WEIGHT_FILE = PARSED_ARGS.output_weight_file

# create directory to the folder
pathToFileDirectory = PATH_TO_DIRECTORY_OF_CONFIG + '/'.join(OUTPUT_WEIGHT_FILE.split('/')[:-1]) + '/'
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

# All paths to ontology extension files are relative to the location of the
# model configuration file.
ontologyExtFiles = map(lambda x: PATH_TO_DIRECTORY_OF_CONFIG + x, ontologyExtFiles)

# Grab all the positive labels
POSITIVE_LABELS = get_recursive_sound_names(CONFIG_DATA["positiveLabels"], "../", ontologyExtFiles)

# If negative labels were provided, then collect them
# Otherwise, assume all examples that are not positive are negative
if CONFIG_DATA["negativeLabels"] is None:
  NEGATIVE_LABELS = None
else:
  # Grab all the negative labels
  NEGATIVE_LABELS = get_recursive_sound_names(CONFIG_DATA["negativeLabels"], "../", ontologyExtFiles)

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


def import_dataframes(dataframe_file_list,
                      positive_label_filter_arr,
                      negative_label_filter_arr,
                      path_to_directory_of_json):
  """
  Iterate through each pickle file and import a subset of the dataframe
  """
  list_of_dataframes = []

  for input_file_dict in dataframe_file_list:

    print "Importing", input_file_dict["path"], "..."

    with open(path_to_directory_of_json + input_file_dict["path"], 'rb') as file_obj:

      # Load the file
      df = pickle.load(file_obj)

      # Filtering the sounds that are exactly 10 seconds
      # Examples should be exactly 10 seconds. Anything else
      # is not a valid input to the model
      df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]

      # Only use examples that have a label in label filter array
      positive_example_select_vector = df['labels_name'].apply(lambda arr: np.any([x.lower() in positive_label_filter_arr for x in arr]))
      positive_examples_df = df.loc[positive_example_select_vector]

      if negative_label_filter_arr is not None:
        negative_example_select_vector = df['labels_name'].apply(lambda arr: np.any([x.lower() in negative_label_filter_arr for x in arr]))
      else:
        negative_example_select_vector = ~positive_example_select_vector
      negative_examples_df = df.loc[negative_example_select_vector]

      # No longer need df after this point
      del df

      positive_subsample = input_file_dict["positiveSubsample"]
      if (positive_examples_df.shape[0] == 0) and (positive_subsample > 0):
        print "No positive examples to subsample from!"
        assert(False)
      else:
        positive_examples_df = subsample_dataframe(positive_examples_df, positive_subsample)

      negative_subsample = input_file_dict["negativeSubsample"]
      if (negative_examples_df.shape[0] == 0) and (negative_subsample > 0):
        print "No negative examples to subsample from!"
        assert(False)
      else:
        negative_examples_df = subsample_dataframe(negative_examples_df, input_file_dict["negativeSubsample"])

      # append to overall list of examples
      list_of_dataframes += [positive_examples_df, negative_examples_df]

  df = pd.concat(list_of_dataframes, ignore_index=True)

  print "Import done."

  return df

DATA_FRAME = import_dataframes(dataframe_file_list=CONFIG_DATA["train"]["inputDataFrames"],
                               positive_label_filter_arr=POSITIVE_LABELS,
                               negative_label_filter_arr=NEGATIVE_LABELS,
                               path_to_directory_of_json=PATH_TO_DIRECTORY_OF_CONFIG)


#############################################################################
        # To Train each class-label change to appropiate label
#############################################################################
LABELS_BINARIZED = pd.DataFrame()
LABELS_BINARIZED[FULL_NAME] = 1.0 * DATA_FRAME['labels_name'].apply(lambda arr: np.any([x.lower() in POSITIVE_LABELS for x in arr]))

#############################################################################
        # print out the number and percentage of each class examples
#############################################################################

print "NUMBER EXAMPLES (TOTAL/POSITIVE/NEGATIVE):", \
      LABELS_BINARIZED.shape[0], "/", \
      (LABELS_BINARIZED[FULL_NAME] == 1).sum(), "/", \
      (LABELS_BINARIZED[FULL_NAME] == 0).sum()

print "PERCENT POSITIVE EXAMPLES:", "{0:.2f}%".format(100*LABELS_BINARIZED[FULL_NAME].mean())



#############################################################################
        # split the data into train and test data
#############################################################################
TRAIN_TEST_SHUFFLE_SPLIT = StratifiedShuffleSplit(n_splits=1,
                                                  test_size=CONFIG_DATA["train"]["validationSplit"],
                                                  random_state=42)

TRAIN_INDICES, TEST_INDICES = next(TRAIN_TEST_SHUFFLE_SPLIT.split(DATA_FRAME, LABELS_BINARIZED))

DF_TRAIN  = DATA_FRAME.iloc[TRAIN_INDICES]
DF_TEST   = DATA_FRAME.iloc[TEST_INDICES]
LABELS_BINARIZED_TRAIN  = LABELS_BINARIZED.iloc[TRAIN_INDICES]
LABELS_BINARIZED_TEST   = LABELS_BINARIZED.iloc[TEST_INDICES]

#############################################################################
        # preprecess the data into required structure
#############################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
X_TEST_STANDARDIZED = X_TEST / 255



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
    print model.summary()
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

#############################################################################
   # assign class weights to ensure balanced datasets during training
#############################################################################

TRAIN_TARGET_POSITIVE_PERCENTAGE = CLF2_TRAIN_TARGET.mean()

print "TRAIN NUMBER EXAMPLES (TOTAL/POSITIVE/NEGATIVE):", \
      CLF2_TRAIN_TARGET.shape[0], "/", \
      (CLF2_TRAIN_TARGET == 1).sum(), "/", \
      (CLF2_TRAIN_TARGET == 0).sum()
print "TRAIN PERCENT POSITIVE EXAMPLES:", "{0:.2f}%".format(100*TRAIN_TARGET_POSITIVE_PERCENTAGE)

if TRAIN_TARGET_POSITIVE_PERCENTAGE > 0.5:
  CLASS_WEIGHT_0 = TRAIN_TARGET_POSITIVE_PERCENTAGE / (1-TRAIN_TARGET_POSITIVE_PERCENTAGE)
  CLASS_WEIGHT_1 = 1
else:
  CLASS_WEIGHT_0 = 1
  CLASS_WEIGHT_1 = (1-TRAIN_TARGET_POSITIVE_PERCENTAGE) / TRAIN_TARGET_POSITIVE_PERCENTAGE

#############################################################################
      # Implementing using the keras usual training techinque
#############################################################################
if CONFIG_DATA["networkCfgJson"] is None:
  MODEL = create_keras_model()
else:
  # load json and create model
  json_file = open(PATH_TO_DIRECTORY_OF_CONFIG + CONFIG_DATA["networkCfgJson"], 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  MODEL = model_from_json(loaded_model_json)
  MODEL.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                metrics=['accuracy'])

MODEL_TRAINING = MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                           epochs=CONFIG_DATA["train"]["epochs"],
                           batch_size=CONFIG_DATA["train"]["batchSize"],
                           class_weight={0: CLASS_WEIGHT_0, 1: CLASS_WEIGHT_1},
                           verbose=1,
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET))

#############################################################################
      # Predict on train and test data
#############################################################################
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
print 'Misclassified number of examples :', MISCLASSIFED_ARRAY.sum()



#############################################################################
        # Print confusion matrix and classification_report
#############################################################################
print CLF2_TEST_TARGET.shape
print '        Confusion Matrix          '
print '============================================'
RESULT = confusion_matrix(CLF2_TEST_TARGET,
                          CLF2_TEST_PREDICTION)
print RESULT



#############################################################################
        # print classification report
#############################################################################
print '                 Classification Report      '
print '============================================'
CL_REPORT = classification_report(CLF2_TEST_TARGET,
                                  CLF2_TEST_PREDICTION)
print CL_REPORT


#############################################################################
        # calculate accuracy and hamming loss
#############################################################################
ACCURACY = accuracy_score(CLF2_TEST_TARGET,
                          CLF2_TEST_PREDICTION)
HL = hamming_loss(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION)
print 'Hamming Loss :', HL
print 'Accuracy :', ACCURACY



#############################################################################
        # save model weights. Change as per the model type
#############################################################################
MODEL.save_weights(PATH_TO_DIRECTORY_OF_CONFIG + OUTPUT_WEIGHT_FILE)

