"""
Detection of Ambient Vs Impact sounds
using the Goertzel frequency components
ie [800, 1600, 2000, 2300]
"""
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed, MaxPooling2D
from tensorflow.compat.v1.keras.optimizers import RMSprop
from . import balancing_dataset_goertzel
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, WOOD_SOUNDS, HUMAN_SOUNDS, NATURE_SOUNDS, DOMESTIC_SOUNDS, TOOLS_SOUNDS
from youtube_audioset import get_recursive_sound_names, get_all_sound_names


###############################################################################
# Description and Help
###############################################################################
DESCRIPTION = "Enter the path for goertzel component files ( .pkl ) "
HELP = "Detects Impact Vs Ambient sounds using Goertzel Frequency components"


###############################################################################
# parse the input arguments given from command line
###############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_for_goertzel_components',
                    '--path_for_goertzel_components', action='store',
                    help=HELP)
RESULT = PARSER.parse_args()


###############################################################################
# get the data of each sounds separately
###############################################################################
DATA_FRAME = balancing_dataset_goertzel.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)


###############################################################################
# getting recursive label names
###############################################################################
AMBIENT_SOUNDS, IMPACT_SOUNDS = get_all_sound_names("./")
EXPLOSION = get_recursive_sound_names(EXPLOSION_SOUNDS, "./")
MOTOR = get_recursive_sound_names(MOTOR_SOUNDS, "./")
WOOD = get_recursive_sound_names(WOOD_SOUNDS, "./")
HUMAN = get_recursive_sound_names(HUMAN_SOUNDS, "./")
NATURE = get_recursive_sound_names(NATURE_SOUNDS, "./")
DOMESTIC = get_recursive_sound_names(DOMESTIC_SOUNDS, "./")
DOMESTIC = get_recursive_sound_names(TOOLS_SOUNDS, "./")


###############################################################################
# Binarize the labels
###############################################################################
NAME_BIN = LabelBinarizer().fit(AMBIENT_SOUNDS + IMPACT_SOUNDS)
LABELS_SPLIT = DATA_FRAME['labels_name'].apply(pd.Series).fillna('None')
LABELS_BINARIZED = NAME_BIN.transform(LABELS_SPLIT[LABELS_SPLIT.columns[0]])
for column in LABELS_SPLIT.columns:
    LABELS_BINARIZED |= NAME_BIN.transform(LABELS_SPLIT[column])
LABELS_BINARIZED = pd.DataFrame(LABELS_BINARIZED, columns=NAME_BIN.classes_)


###############################################################################
# Shuffle the data
###############################################################################
DATA_FRAME, LABELS_BINARIZED = shuffle(DATA_FRAME, LABELS_BINARIZED, random_state=20)


###############################################################################
# print out the shape and percentage of sounds
###############################################################################
print('Binarized labels shape :', LABELS_BINARIZED.shape)
print("Percentage Impact Sounds:", (LABELS_BINARIZED[IMPACT_SOUNDS].sum(axis=1) > 0).mean())
print("Percentage Ambient Sounds:", (LABELS_BINARIZED[IMPACT_SOUNDS].sum(axis=1) > 0).mean())


###############################################################################
# split up the data into train and test
###############################################################################
DF_TRAIN, DF_TEST, LABELS_BINARIZED_TRAIN, LABELS_BINARIZED_TEST = train_test_split(DATA_FRAME,
                                                                                    LABELS_BINARIZED,
                                                                                    test_size=0.33,
                                                                                    random_state=42
                                                                                   )


###############################################################################
# Create the time distributed model
###############################################################################
def create_keras_model():
    """
    Time Distributed sequential model
    for parameter sharing of each second
    """
    model = Sequential()
    model.add(TimeDistributed(MaxPooling1D(80), input_shape=(10, 8000, 4)))
    model.add(TimeDistributed(Conv1D(20,
                                     kernel_size=100,
                                     strides=50,
                                     activation='relu',
                                     padding='same')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.add(MaxPooling2D((10, 1)))
    model.add(Flatten())
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(1e-3), metrics=['accuracy'])
    return model


###############################################################################
# define the target labels for test and train
###############################################################################
CLF1_TEST_TARGET = LABELS_BINARIZED_TEST.loc[:, IMPACT_SOUNDS].any(axis=1)
CLF1_TRAIN_TARGET_MINI = np.array(LABELS_BINARIZED_TRAIN.loc[:, IMPACT_SOUNDS].any(axis=1),
                                  dtype=float)


###############################################################################
# Create the Model
###############################################################################
MODEL = create_keras_model()
CLF1_TRAIN_MINI = []
CLF1_TEST = []
print('Reading Test files ..!')


###############################################################################
# Read all the test data first
###############################################################################
for each_emb, each_wav in zip(DF_TEST['features'].tolist(), DF_TEST["wav_file"].tolist()):
    # Read all the files that are splitted as test in the path directory specified
    try:
        CLF1_TEST.append(np.dstack((each_emb[0].reshape((10, 8000)),
                                    each_emb[1].reshape((10, 8000)),
                                    each_emb[2].reshape((10, 8000)),
                                    each_emb[3].reshape((10, 8000)))))

    # except any error then remove that file manually and then run the process again
    except:
        print('Test Pickling Error: ', each_wav)


###############################################################################
        # reshaping test data and applying normalization
###############################################################################
print(np.array(CLF1_TEST).shape)
CLF1_TEST = np.array(CLF1_TEST).reshape((-1, 10, 8000, 4))
CLF1_TEST = CLF1_TEST / np.linalg.norm(CLF1_TEST)


###############################################################################
# Reading Training Files
###############################################################################
print("Reading Training files..!!")
for each_emb, each_wav in zip(DF_TRAIN['features'].tolist(), DF_TRAIN["wav_file"].tolist()):
    # Read all the files that are splitted as train in the path directory specified
    try:
        CLF1_TRAIN_MINI.append(np.dstack((each_emb[0].reshape((10, 8000)),
                                          each_emb[1].reshape((10, 8000)),
                                          each_emb[2].reshape((10, 8000)),
                                          each_emb[3].reshape((10, 8000)))))
    except:
        print('Train pickling Error ', each_wav)


###############################################################################
        # Reshaping the traininig data and applying normalization
###############################################################################
CLF1_TRAIN_MINI = np.array(CLF1_TRAIN_MINI).reshape((-1, 10, 8000, 4))
CLF1_TRAIN_MINI = CLF1_TRAIN_MINI / np.linalg.norm(CLF1_TRAIN_MINI)

# start training on model
MODEL.fit(CLF1_TRAIN_MINI,
          CLF1_TRAIN_TARGET_MINI,
          epochs=100,
          verbose=1,
          validation_data=(CLF1_TEST, CLF1_TEST_TARGET))


###############################################################################
# predict out of the model (Change Decision threshold by uncommenting)
###############################################################################
CLF1_TRAIN_PREDICTION_PROB = MODEL.predict(CLF1_TRAIN_MINI).ravel()
CLF1_TRAIN_PREDICTION = MODEL.predict(CLF1_TRAIN_MINI).ravel().round()
# clf1_train_prediction = np.array([0 if i<0.45 else 1 for i in clf1_train_prediction_prob])


###############################################################################
# Predict out the test data. (Change Decision threshold by uncommenting)
###############################################################################
CLF1_TEST_PREDICTION_PROB = MODEL.predict(CLF1_TEST).ravel()
CLF1_TEST_PREDICTION = MODEL.predict(CLF1_TEST).ravel().round()
# clf1_test_prediction = np.array([0 if i<0.22 else 1 for i in clf1_test_prediction_prob])


###############################################################################
# print train and test acuuracy
###############################################################################
print("Train Accuracy:", (CLF1_TRAIN_PREDICTION == CLF1_TRAIN_TARGET_MINI).mean())
print("Test Accuracy:", (CLF1_TEST_PREDICTION == CLF1_TEST_TARGET).mean())


###############################################################################
# print out the confusion matrix for train data
###############################################################################
CLF1_CONF_TRAIN_MAT = pd.crosstab(CLF1_TRAIN_TARGET_MINI, CLF1_TRAIN_PREDICTION, margins=True)
print("Training Precision and recall for Keras model")
print('=============================================')
print("Train Precision:", CLF1_CONF_TRAIN_MAT[True][True] / float(CLF1_CONF_TRAIN_MAT[True]['All']))
print("Train Recall:", CLF1_CONF_TRAIN_MAT[True][True] / float(CLF1_CONF_TRAIN_MAT['All'][True]))
print("Train Accuracy:", (CLF1_TRAIN_PREDICTION == CLF1_TRAIN_TARGET_MINI).mean())
print(CLF1_CONF_TRAIN_MAT)


###############################################################################
# print out the confusion matrix for test data
###############################################################################
CLF1_CONF_TEST_MAT = pd.crosstab(CLF1_TEST_TARGET, CLF1_TEST_PREDICTION, margins=True)
print("Testing Precision and recall for Keras model")
print('=============================================')
print("Test Precision:", CLF1_CONF_TEST_MAT[True][True] / float(CLF1_CONF_TEST_MAT[True]['All']))
print("Test Recall:", CLF1_CONF_TEST_MAT[True][True] / float(CLF1_CONF_TEST_MAT['All'][True]))
print("Test Accuracy:", (CLF1_TEST_PREDICTION == CLF1_TEST_TARGET).mean())
print(CLF1_CONF_TEST_MAT)


###############################################################################
# calculate the f1 score and print it out
###############################################################################
F1_SCORE = metrics.f1_score(CLF1_TEST_TARGET, CLF1_TEST_PREDICTION)
print('F1 score is  : ', F1_SCORE)


###############################################################################
# save the model
###############################################################################
MODEL.save_weights('Goertzel_model_8k_weights_time.h5')


###############################################################################
# save the model weights for each layer separately
###############################################################################
WEIGHTS_LIST = []
for layer in MODEL.layers:
    WEIGHTS_LIST.append(layer.get_weights())

with open("weights_goertzel_model_layers.pkl", "wb") as f:
    pickle.dump(WEIGHTS_LIST, f)
