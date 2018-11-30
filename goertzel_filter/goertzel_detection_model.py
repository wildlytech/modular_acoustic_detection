"""
Detection of Ambient Vs Impact sounds
using the Goertzel frequency components
ie [800, 1600, 200, 2300]
"""
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed, MaxPooling2D
from keras.layers.core import Lambda
from keras.optimizers import RMSprop
from keras_tqdm import TQDMNotebookCallback
from keras.layers.core import Dropout
from youtube_audioset import explosion_sounds, motor_sounds, wood_sounds, human_sounds, nature_sounds, Wild_animals,domestic_sounds
from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names
import balancing_dataset
import frequency_component_files


DESCRIPTION = "Enter the path for goertzel component files ( .pkl ) "
HELP = "Detects Impact Vs Ambient sounds using Goertzel Frequency components"

#parse the input arguments given from command line
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_for_goertzel_components',
                    '--path_for_goertzel_components', action='store',
                    help=HELP)
RESULT = PARSER.parse_args()

#get the data of each sounds seperately and then concat all sounds to get balanced data
MOT, HUM, WOD, EXP, DOM, TOOLS, WILD, NAT = frequency_component_files.get_req_sounds(RESULT.path_for_goertzel_components)


# Try to balance the number of examples.
# Here we need to balance as Impact Vs Ambient , but not as multilabel sounds
DATA_FRAME = pd.concat([MOT[:3000],
                        HUM[:1700],
                        WOD[:500],
                        EXP[:1200],
                        DOM[:1100],
                        TOOLS[:1500],
                        WILD[:1000],
                        NAT[:8000]],
                       ignore_index=True)


# execute the labels binarized by importing the youtube_audioset function
AMBIENT_SOUNDS, IMPACT_SOUNDS = get_all_sound_names()
EXPLOSION = get_recursive_sound_names(explosion_sounds)
MOTOR = get_recursive_sound_names(motor_sounds)
WOOD = get_recursive_sound_names(wood_sounds)
HUMAN = get_recursive_sound_names(human_sounds)
NATURE = get_recursive_sound_names(nature_sounds)
DOMESTIC = get_recursive_sound_names(domestic_sounds)

#Binarize the labels
NAME_BIN = LabelBinarizer().fit(AMBIENT_SOUNDS + IMPACT_SOUNDS)
LABELS_SPLIT = DATA_FRAME['labels_name'].apply(pd.Series).fillna('None')
LABELS_BINARIZED = NAME_BIN.transform(LABELS_SPLIT[LABELS_SPLIT.columns[0]])
for column in LABELS_SPLIT.columns:
    LABELS_BINARIZED |= NAME_BIN.transform(LABELS_SPLIT[column])
LABELS_BINARIZED = pd.DataFrame(LABELS_BINARIZED, columns=NAME_BIN.classes_)

#Shuffle the data
DATA_FRAME, LABELS_BINARIZED = shuffle(DATA_FRAME, LABELS_BINARIZED, random_state=20)

#print out the shape and percentage of sounds
print 'Binarized labels shape :', LABELS_BINARIZED.shape
print "Percentage Impact Sounds:", (LABELS_BINARIZED[IMPACT_SOUNDS].sum(axis=1) > 0).mean()
print "Percentage Ambient Sounds:", (LABELS_BINARIZED[IMPACT_SOUNDS].sum(axis=1) > 0).mean()

# split up the data into train and test
DF_TRAIN, DF_TEST, LABELS_BINARIZED_TRAIN, LABELS_BINARIZED_TEST = train_test_split(DATA_FRAME,
                                                                                    LABELS_BINARIZED,
                                                                                    test_size=0.1,
                                                                                    random_state=42
                                                                                   )

# Create the time distributed model
def create_keras_model():
    """
    Time Distributed sequential model
    for parameter sharing of each second
    """
    model = Sequential()
    model.add(TimeDistributed(Conv1D(100,
                                     kernel_size=200,
                                     strides=100,
                                     activation='relu',
                                     padding='same'),
                              input_shape=(10, 8000, 4)))
    model.add(TimeDistributed(Conv1D(100,
                                     kernel_size=4,
                                     activation='relu',
                                     padding='same')))
    model.add(TimeDistributed(MaxPooling1D(80)))
    model.add(TimeDistributed(Dense(60, activation='relu')))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.add(MaxPooling2D((10, 1)))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(1e-3), metrics=['accuracy'])
    return model

# define the target labels for test and train
CLF1_TEST_TARGET = LABELS_BINARIZED_TEST.loc[:, IMPACT_SOUNDS].any(axis=1)
CLF1_TRAIN_TARGET_MINI = np.array(LABELS_BINARIZED_TRAIN.loc[:, IMPACT_SOUNDS].any(axis=1),
                                  dtype=float)

#Train the model
MODEL = create_keras_model()
CLF1_TRAIN_MINI = []
CLF1_TEST = []
print 'Reading Test files ..!'

# Read all the test data first
for wav_file in DF_TEST['wav_file'].tolist():
    # Read all the files that are splitted as test in the path directory specified
    try:
        with open(RESULT.path_for_goertzel_components + wav_file[:11]+'.pkl', 'rb') as f:
            arb_wav = pickle.load(f)
        CLF1_TEST.append(np.dstack((arb_wav[0].reshape((10, 8000)),
                                    arb_wav[1].reshape((10, 8000)),
                                    arb_wav[2].reshape((10, 8000)),
                                    arb_wav[3].reshape((10, 8000)))))

    #except any error then remove that file manually and then run the process again
    except :
        print 'Test Pickling Error'
        print wav_file

#reshaping test data and applying normalization
print np.array(CLF1_TEST).shape
CLF1_TEST = np.array(CLF1_TEST).reshape((-1, 10, 8000, 4))
CLF1_TEST = CLF1_TEST / np.linalg.norm(CLF1_TEST)

print "Reading Training files..!!"
for wav in DF_TRAIN['wav_file'].tolist():
    # Read all the files that are splitted as train in the path directory specified
    try:
        with open(RESULT.path_for_goertzel_components + wav[:11]+'.pkl', 'rb') as f:
            arb_wav = pickle.load(f)
        CLF1_TRAIN_MINI.append(np.dstack((arb_wav[0].reshape((10, 8000)),
                                          arb_wav[1].reshape((10, 8000)),
                                          arb_wav[2].reshape((10, 8000)),
                                          arb_wav[3].reshape((10, 8000)))))
    except:
        print 'Train pickling Error '
        print wav

# reshaping the traininig data and applying normalization
CLF1_TRAIN_MINI = np.array(CLF1_TRAIN_MINI).reshape((-1, 10, 8000, 4))
CLF1_TRAIN_MINI = CLF1_TRAIN_MINI/np.linalg.norm(CLF1_TRAIN_MINI)

#start training on model
MODEL.fit(CLF1_TRAIN_MINI,
          CLF1_TRAIN_TARGET_MINI,
          epochs=30,
          verbose=1,
          validation_data=(CLF1_TEST, CLF1_TEST_TARGET))


#predict out of the model

CLF1_TRAIN_PREDICTION_PROB = MODEL.predict(CLF1_TRAIN_MINI).ravel()
CLF1_TRAIN_PREDICTION = MODEL.predict(CLF1_TRAIN_MINI).ravel().round()
# clf1_train_prediction = np.array([0 if i<0.45 else 1 for i in clf1_train_prediction_prob])

CLF1_TEST_PREDICTION_PROB = MODEL.predict(CLF1_TEST).ravel()
CLF1_TEST_PREDICTION = MODEL.predict(CLF1_TEST).ravel().round()
# clf1_test_prediction = np.array([0 if i<0.22 else 1 for i in clf1_test_prediction_prob])
print CLF1_TEST_PREDICTION_PROB[CLF1_TEST_PREDICTION_PROB < 0.3]

# print train and test acuuracy
print "Train Accuracy:", (CLF1_TRAIN_PREDICTION == CLF1_TRAIN_TARGET_MINI).mean()
print "Test Accuracy:", (CLF1_TEST_PREDICTION == CLF1_TEST_TARGET).mean()

#print out the confusion matrix for train data
CLF1_CONF_TRAIN_MAT = pd.crosstab(CLF1_TRAIN_TARGET_MINI, CLF1_TRAIN_PREDICTION, margins=True)
print "Training Precision and recall for Keras model"
print '============================================='
print "Train Precision:", CLF1_CONF_TRAIN_MAT[True][True] / float(CLF1_CONF_TRAIN_MAT[True]['All'])
print "Train Recall:", CLF1_CONF_TRAIN_MAT[True][True] / float(CLF1_CONF_TRAIN_MAT['All'][True])
print "Train Accuracy:", (CLF1_TRAIN_PREDICTION == CLF1_TRAIN_TARGET_MINI).mean()
print CLF1_CONF_TRAIN_MAT


#print out the confusion matrix for test data
CLF1_CONF_TEST_MAT = pd.crosstab(CLF1_TEST_TARGET, CLF1_TEST_PREDICTION, margins=True)
print "Testing Precision and recall for Keras model"
print '============================================='
print "Test Precision:", CLF1_CONF_TEST_MAT[True][True] / float(CLF1_CONF_TEST_MAT[True]['All'])
print "Test Recall:", CLF1_CONF_TEST_MAT[True][True] / float(CLF1_CONF_TEST_MAT['All'][True])
print "Test Accuracy:", (CLF1_TEST_PREDICTION == CLF1_TEST_TARGET).mean()
print CLF1_CONF_TEST_MAT

# calculate the f1 score and print it out
F1_SCORE = metrics.f1_score(CLF1_TEST_TARGET, CLF1_TEST_PREDICTION)
print 'F1 score is  : ', F1_SCORE

#save the model
MODEL.save_weights('Goertzel_model_8k_weights_time.h5')
