"""
Traning a Binary Relevance Model
"""
# Import the necessary functions and libraries
import argparse
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

DF_TRAIN, DF_TEST = \
    import_dataframes(dataframe_file_list=CONFIG_DATA["train"]["inputDataFrames"],
                      positive_label_filter_arr=POSITIVE_LABELS,
                      negative_label_filter_arr=NEGATIVE_LABELS,
                      validation_split=CONFIG_DATA["train"]["validationSplit"])


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
    print(model.summary())
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
if CONFIG_DATA["networkCfgJson"] is None:
    MODEL = create_keras_model()
else:
    # load json and create model
    json_file = open(CONFIG_DATA["networkCfgJson"], 'r')
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
MODEL.save_weights(OUTPUT_WEIGHT_FILE)
