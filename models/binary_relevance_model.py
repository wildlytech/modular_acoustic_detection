"""
Traning a Binary Relevance Model
"""
# Import the necessary functions and libraries
import argparse
import json
from tensorflow.compat.v1.keras.models import model_from_json
from tensorflow.compat.v1.keras.optimizers import Adam
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from tensorflow.keras.callbacks import ModelCheckpoint
from youtube_audioset import get_recursive_sound_names
from keras_balanced_batch_generator import make_generator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import BinaryCrossentropy
from .preprocess_utils import import_dataframes, get_select_vector

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
ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
PARSED_ARGS = ARGUMENT_PARSER.parse_args()

#############################################################################
# Parse the arguments
#############################################################################

with open(PARSED_ARGS.model_cfg_json) as json_file_obj:
    CONFIG_DATA = json.load(json_file_obj)

FULL_NAME = CONFIG_DATA["aggregatePositiveLabelName"] + 'vs' + CONFIG_DATA["aggregateNegativeLabelName"]

# Set default output weight file name
OUTPUT_WEIGHT_FILE = CONFIG_DATA["train"]["outputWeightFile"]

# create directory to the folder
pathToFileDirectory = "/".join(OUTPUT_WEIGHT_FILE.split('/')[:-1]) + '/'
if not os.path.exists(pathToFileDirectory):
    os.makedirs(pathToFileDirectory)

#############################################################################
# Get all sound names
#############################################################################

# Model training only supports using audio set as main ontology
assert (CONFIG_DATA["ontology"]["useYoutubeAudioSet"])

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

print("PERCENT POSITIVE EXAMPLES:",
      "{0:.2f}%".format(100.0 * TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES / TOTAL_TRAIN_TEST_EXAMPLES))

#############################################################################
# preprocess the data into required structure
#############################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
X_TEST_STANDARDIZED = X_TEST / 255


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
print("TRAIN PERCENT POSITIVE EXAMPLES:", "{0:.2f}%".format(100 * TRAIN_TARGET_POSITIVE_PERCENTAGE))

if TRAIN_TARGET_POSITIVE_PERCENTAGE > 0.5:
    CLASS_WEIGHT_0 = TRAIN_TARGET_POSITIVE_PERCENTAGE / (1 - TRAIN_TARGET_POSITIVE_PERCENTAGE)
    CLASS_WEIGHT_1 = 1
else:
    CLASS_WEIGHT_0 = 1
    CLASS_WEIGHT_1 = (1 - TRAIN_TARGET_POSITIVE_PERCENTAGE) / TRAIN_TARGET_POSITIVE_PERCENTAGE

#############################################################################
# Implementing using the keras usual training technique
#############################################################################
checkpoint_path = "checkpoint/cp.ckpt"
callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

training_generator = make_generator(
    CLF2_TRAIN, to_categorical(CLF2_TRAIN_TARGET), batch_size=CONFIG_DATA["train"]["batchSize"], categorical=False,
    seed=42)

# load json and create model
json_file = open(CONFIG_DATA["networkCfgJson"], 'r')
loaded_model_json = json_file.read()
json_file.close()
MODEL = model_from_json(loaded_model_json)
MODEL.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, epsilon=1e-8),
              metrics=['accuracy'])

steps_per_epoch = len(CLF2_TRAIN) // CONFIG_DATA["train"]["batchSize"]
MODEL_TRAINING = MODEL.fit(training_generator, shuffle=True,
                           epochs=CONFIG_DATA["train"]["epochs"],
                           steps_per_epoch=steps_per_epoch,
                           callbacks=[callback],
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET.reshape(-1)), verbose=1)

print("Load model weights from checkpoint: ")
MODEL.load_weights(checkpoint_path)
print("Model Loaded")

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

print("******* FINAL VAL LOSS *******")
bce = BinaryCrossentropy()
print(bce(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION_PROB).numpy())
print("********************************")

#############################################################################
# save model weights. Change as per the model type
#############################################################################
MODEL.save_weights(OUTPUT_WEIGHT_FILE)
