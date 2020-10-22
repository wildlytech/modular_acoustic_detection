"""
Training a Binary Output model.
Impact Sounds = 1
Ambient SOunds = 0
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.optimizers import RMSprop
from youtube_audioset import get_recursive_sound_names, get_all_sound_names
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, \
                             WOOD_SOUNDS, HUMAN_SOUNDS, NATURE_SOUNDS, AMBIENT_SOUNDS, IMPACT_SOUNDS
import balancing_dataset






########################################################################
            # get all the sounds
########################################################################
AMBIENT_SOUNDS, IMPACT_SOUNDS = get_all_sound_names("./")
explosion_sounds = get_recursive_sound_names(EXPLOSION_SOUNDS, "./")
motor_sounds = get_recursive_sound_names(MOTOR_SOUNDS, "./")
wood_sounds = get_recursive_sound_names(WOOD_SOUNDS, "./")
human_sounds = get_recursive_sound_names(HUMAN_SOUNDS, "./")
nature_sounds = get_recursive_sound_names(NATURE_SOUNDS, "./")



########################################################################
            # Read the balanced data
            # Note that this is binary classification.
            # Balancing must be  [ Ambient ] vs  [ Impact ]
########################################################################
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)



########################################################################
            # Binarize the labels
########################################################################
NAME_BIN = LabelBinarizer().fit(AMBIENT_SOUNDS + IMPACT_SOUNDS)
LABELS_SPLIT = DATA_FRAME['labels_name'].apply(pd.Series).fillna('None')
LABELS_BINARIZED = NAME_BIN.transform(LABELS_SPLIT[LABELS_SPLIT.columns[0]])
for column in LABELS_SPLIT.columns:
    LABELS_BINARIZED |= NAME_BIN.transform(LABELS_SPLIT[column])
LABELS_BINARIZED = pd.DataFrame(LABELS_BINARIZED, columns=NAME_BIN.classes_)




########################################################################
            # print the percentage of Impact and Ambinet sounds
########################################################################
print("Percentage Impact Sounds:", (LABELS_BINARIZED[IMPACT_SOUNDS].sum(axis=1) > 0).mean())
print("Percentage Ambient Sounds:", (LABELS_BINARIZED[AMBIENT_SOUNDS].sum(axis=1) > 0).mean())



########################################################################
            # Filter out the sounds that are having 10 seconds duration.
########################################################################
DF_FILTERED = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
LABELS_FILTERED = LABELS_BINARIZED.loc[DF_FILTERED.index, :]



########################################################################
            # Split the data into train and test
########################################################################
DF_TRAIN, DF_TEST, LABELS_BINARIZED_TRAIN, LABELS_BINARIZED_TEST = train_test_split(DF_FILTERED, LABELS_FILTERED,
                                                                                    test_size=0.33, random_state=42,
                                                                                    stratify=LABELS_FILTERED.any(axis=1)*1)




########################################################################
            # Setting the target as "feature"
########################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
X_TEST_STANDARDIZED = X_TEST / 255
Y_TRAIN = (LABELS_BINARIZED_TRAIN[IMPACT_SOUNDS].any(axis=1)*1).values
Y_TEST = (LABELS_BINARIZED_TEST[IMPACT_SOUNDS].any(axis=1)*1).values




########################################################################
            # Print the percentage of each sounds in whole data
########################################################################
print(LABELS_FILTERED.loc[:, explosion_sounds].any(axis=1).mean())
print(LABELS_FILTERED.loc[:, motor_sounds].any(axis=1).mean())
print(LABELS_FILTERED.loc[:, wood_sounds].any(axis=1).mean())
print(LABELS_FILTERED.loc[:, human_sounds].any(axis=1).mean())
print(LABELS_FILTERED.loc[:, nature_sounds].any(axis=1).mean())
print(LABELS_FILTERED.loc[:, IMPACT_SOUNDS].any(axis=1).mean())



########################################################################
            # Try experimenting with Logistic regression algorithm
########################################################################
CLF1_ = LogisticRegression(max_iter=1000)
CLF1_TRAIN = X_TRAIN
CLF1_TEST = X_TEST



########################################################################
        # Assign the Labels
        # Impact sounds( target sounds) as  1's
        # and ambient sounds as 0's
########################################################################
CLF1_TRAIN_TARGET = LABELS_BINARIZED_TRAIN.loc[:, IMPACT_SOUNDS].any(axis=1)
CLF1_TEST_TARGET = LABELS_BINARIZED_TEST.loc[:, IMPACT_SOUNDS].any(axis=1)



########################################################################
        # fit the train data to LR model
########################################################################
print("Trainging Logistic Regression Model..")
CLF1_.fit(CLF1_TRAIN, CLF1_TRAIN_TARGET)


########################################################################
        # Predict on the trained LR model
########################################################################
CLF1_TRAIN_PREDICTION = CLF1_.predict(CLF1_TRAIN)
CLF1_TEST_PREDICTION = CLF1_.predict(CLF1_TEST)
CLF1_TEST_PREDICTION_PROB = CLF1_.predict_proba(CLF1_TEST)[:, 1]


########################################################################
        # Print out the confusion matrix for Train data
########################################################################
CLF1_CONF_TRAIN_MAT = pd.crosstab(CLF1_TRAIN_TARGET, CLF1_TRAIN_PREDICTION, margins=True)
print('Train precsion and recall for Logistic regression')
print('=============================================')
print("Train Precision:", CLF1_CONF_TRAIN_MAT[True][True] / float(CLF1_CONF_TRAIN_MAT[True]['All']))
print("Train Recall:", CLF1_CONF_TRAIN_MAT[True][True] / float(CLF1_CONF_TRAIN_MAT['All'][True]))
print("Train Accuracy:", (CLF1_TRAIN_PREDICTION == CLF1_TRAIN_TARGET).mean())
print(CLF1_CONF_TRAIN_MAT)



########################################################################
        # Print out the confusion matrix for test data
########################################################################
CLF1_CONF_TEST_MAT = pd.crosstab(CLF1_TEST_TARGET, CLF1_TEST_PREDICTION, margins=True)
print('Test precsion and recall for Logistic regression')
print('=============================================')
print("Test Precision:", CLF1_CONF_TEST_MAT[True][True] / float(CLF1_CONF_TEST_MAT[True]['All']))
print("Test Recall:", CLF1_CONF_TEST_MAT[True][True] / float(CLF1_CONF_TEST_MAT['All'][True]))
print("Test Accuracy:", (CLF1_TEST_PREDICTION == CLF1_TEST_TARGET).mean())
print(CLF1_CONF_TEST_MAT)




########################################################################
        # create the keras neural netwrok model
########################################################################
def create_keras_model():
    """
    Create a Model
    """
    model = Sequential()
    model.add(Conv1D(40, input_shape=(1280, 1), kernel_size=128,
                     strides=128, activation='relu', padding='same'))
    model.add(Conv1D(100, kernel_size=3, activation='relu', padding='same'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


########################################################################
        # Assign the train and test data and
        # reshape so as to align it to neural network model's input.
########################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TEST = X_TEST.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET = LABELS_BINARIZED_TRAIN.loc[:, IMPACT_SOUNDS].any(axis=1)
CLF2_TEST_TARGET = LABELS_BINARIZED_TEST.loc[:, IMPACT_SOUNDS].any(axis=1)


########################################################################
        # call the model and start training
########################################################################
MODEL = create_keras_model()
MODEL_TRAINING = MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                           epochs=100, batch_size=500, verbose=True,
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET))


########################################################################
        # Predict on test and train data using the tranined weights
########################################################################
CLF2_TRAIN_PREDICTION = MODEL.predict(CLF2_TRAIN).ravel().round()
CLF2_TEST_PREDICTION = MODEL.predict(CLF2_TEST).ravel().round()
CLF2_TEST_PREDICTION_PROB = MODEL.predict(CLF2_TEST).ravel()


########################################################################
        # Accuracy of Train and test
########################################################################
print("Train Accuracy:", (CLF2_TRAIN_PREDICTION == CLF2_TRAIN_TARGET).mean())
print("Test Accuracy:", (CLF2_TEST_PREDICTION == CLF2_TEST_TARGET).mean())


########################################################################
        # print out the confusion matrix for train data
########################################################################
CLF2_CONF_TRAIN_MAT = pd.crosstab(CLF2_TRAIN_TARGET, CLF2_TRAIN_PREDICTION, margins=True)
print("Training Precision and recall for Keras model")
print('=============================================')
print("Train Precision:", CLF2_CONF_TRAIN_MAT[True][True] / float(CLF2_CONF_TRAIN_MAT[True]['All']))
print("Train Recall:", CLF2_CONF_TRAIN_MAT[True][True] / float(CLF2_CONF_TRAIN_MAT['All'][True]))
print("Train Accuracy:", (CLF2_TRAIN_PREDICTION == CLF2_TRAIN_TARGET).mean())
print(CLF2_CONF_TRAIN_MAT)



########################################################################
        # print out the confusion matrix for test data
########################################################################
CLF2_CONF_TEST_MAT = pd.crosstab(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION, margins=True)
print("Testing Precision and recall for Keras model")
print('=============================================')
print("Test Precision:", CLF2_CONF_TEST_MAT[True][True] / float(CLF2_CONF_TEST_MAT[True]['All']))
print("Test Recall:", CLF2_CONF_TEST_MAT[True][True] / float(CLF2_CONF_TEST_MAT['All'][True]))
print("Test Accuracy:", (CLF2_TEST_PREDICTION == CLF2_TEST_TARGET).mean())
print(CLF2_CONF_TEST_MAT)
