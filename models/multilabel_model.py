"""
Training a Multilabel Model
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from tensorflow.compat.v1.keras.models import model_from_json
from tensorflow.compat.v1.keras.optimizers import Adam
from youtube_audioset import get_recursive_sound_names
import os
import json
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_balanced_batch_generator import make_generator
from tensorflow.keras.losses import BinaryCrossentropy
from .preprocess_utils import import_dataframes, get_select_vector


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


DF_TRAIN, DF_TEST = import_dataframes(dataframe_file_list=config["train"]["inputDataFrames"],
                                      positive_label_filter_arr=pos_sounds,
                                      negative_label_filter_arr=neg_sounds,
                                      validation_split=config["train"]["validationSplit"],
                                      model="multilabel")

LABELS_BINARIZED_TRAIN = pd.DataFrame()
LABELS_BINARIZED_TEST = pd.DataFrame()
for key in pos_sounds.keys():
    FULL_NAME = key
    POSITIVE_LABELS = pos_sounds[key]
    LABELS_BINARIZED_TRAIN[FULL_NAME] = 1.0 * get_select_vector(DF_TRAIN, POSITIVE_LABELS)

    LABELS_BINARIZED_TEST[FULL_NAME] = 1.0 * get_select_vector(DF_TEST, POSITIVE_LABELS)

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
# reshaping the train and test data so as to align with input for model
########################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TEST = X_TEST.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET = LABELS_BINARIZED_TRAIN
CLF2_TEST_TARGET = LABELS_BINARIZED_TEST

########################################################################
# Implementing & Training the keras model
########################################################################
json_file = open(config["networkCfgJson"], 'r')
loaded_model_json = json_file.read()
json_file.close()

MODEL = model_from_json(loaded_model_json)
MODEL.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5, epsilon=1e-8),
              metrics=['accuracy'])

epochs = config["train"]["epochs"]
batch_size = config["train"]["batchSize"]

checkpoint_path = "checkpoint/cp.ckpt"
callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    save_weights_only=True,
    verbose=1

)

training_generator = make_generator(
    CLF2_TRAIN, CLF2_TRAIN_TARGET.values, batch_size=batch_size, categorical=True, seed=42)

steps_per_epoch = len(CLF2_TRAIN) // batch_size
MODEL_TRAINING = MODEL.fit(training_generator, shuffle=True,
                           epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[callback],
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET.values), verbose=1)
print("Load model weights from checkpoint: ")
MODEL.load_weights(checkpoint_path)
print("Model Loaded")

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

print("******* FINAL VAL LOSS *******")
bce = BinaryCrossentropy()
print(bce(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION_PROB).numpy())
print("********************************")
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
