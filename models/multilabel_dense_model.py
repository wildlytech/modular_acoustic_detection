"""
Traning a Mulit-label Model
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
sys.path.insert(0, "../")
from youtube_audioset import get_recursive_sound_names, get_all_sound_names
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, WOOD_SOUNDS, \
                             HUMAN_SOUNDS, NATURE_SOUNDS, DOMESTIC_SOUNDS, TOOLS_SOUNDS
import balancing_dataset




########################################################################
          # Get all sound names
########################################################################
explosion_sounds = get_recursive_sound_names(EXPLOSION_SOUNDS, "../")
motor_sounds = get_recursive_sound_names(MOTOR_SOUNDS, "../")
wood_sounds = get_recursive_sound_names(WOOD_SOUNDS, "../")
human_sounds = get_recursive_sound_names(HUMAN_SOUNDS, "../")
nature_sounds = get_recursive_sound_names(NATURE_SOUNDS, "../")
domestic_sounds = get_recursive_sound_names(DOMESTIC_SOUNDS, "../")
tools = get_recursive_sound_names(TOOLS_SOUNDS, "../")
#wild_animals=get_recursive_sound_names(Wild_animals)




########################################################################
      # Importing balanced data from the function.
      # Including audiomoth annotated files for training
########################################################################
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)



########################################################################
      # Different classes of sounds.
      # You can increase the class by adding the necesssary sounds of that class
########################################################################
ALL_SOUND_NAMES = ['Motor_sound', 'Explosion_sound', 'Human_sound',
                   'Nature_sound', 'Domestic_animals', 'Tools']
ALL_SOUND_LIST = explosion_sounds + motor_sounds + human_sounds + \
                 nature_sounds + domestic_sounds + tools




########################################################################
      # Map all the sounds into their respective classes
      # Include the similar column if a new class label is to be added
########################################################################
DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Motor_sound' if x  in motor_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Explosion_sound' if x  in explosion_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Nature_sound' if x  in nature_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Human_sound' if x  in human_sounds else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Wood_sound' if x  in wood_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Domestic_animals' if x  in domestic_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Tools' if x  in tools else x for x in arr])
# DATA_FRAME['labels_new']=DATA_FRAME['labels_new'].apply(lambda arr: ['Wild_animals' if x  in Wild_animals else x for x in arr])



########################################################################
      # Binarize the labels. Its a Multi-label binarizer
########################################################################
NAME_BIN = MultiLabelBinarizer().fit(DATA_FRAME['labels_new'])
LABELS_BINARIZED = NAME_BIN.transform(DATA_FRAME['labels_new'])
LABELS_BINARIZED_ALL = pd.DataFrame(LABELS_BINARIZED, columns=NAME_BIN.classes_)
LABELS_BINARIZED = LABELS_BINARIZED_ALL[ALL_SOUND_NAMES]


########################################################################
      # print out the number and percenatge of each class examples
########################################################################
print LABELS_BINARIZED.mean()



########################################################################
      # Filtering the sounds that are exactly 10 seconds
########################################################################
DF_FILTERED = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
LABELS_FILTERED = LABELS_BINARIZED.loc[DF_FILTERED.index, :]
#print(LABELS_FILTERED)



########################################################################
      # split the data into train and test data
########################################################################
DF_TRAIN, DF_TEST, LABELS_BINARIZED_TRAIN, LABELS_BINARIZED_TEST = train_test_split(DF_FILTERED, LABELS_FILTERED,
                                                                                    test_size=0.33, random_state=42)



########################################################################
      # preprecess the data into required structure
########################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
X_TEST_STANDARDIZED = X_TEST / 255



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
    model.add(MaxPooling1D(10))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Flatten())
    print model.summary()
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
MODEL = create_keras_model()
MODEL_TRAINING = MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                           epochs=100, batch_size=500, verbose=1,
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET))




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
print 'Misclassified number of examples :', DF_TEST[MISCLASSIFIED_EXAMPLES].shape[0]



########################################################################
      # Print confusion matrix and classification_report
########################################################################
print CLF2_TEST_TARGET.values.argmax(axis=1).shape
print '        Confusion Matrix          '
print '============================================'
RESULT = confusion_matrix(CLF2_TEST_TARGET.values.argmax(axis=1),
                          CLF2_TEST_PREDICTION.argmax(axis=1))
print RESULT
print '        Classification Report      '
print '============================================'
CL_REPORT = classification_report(CLF2_TEST_TARGET.values.argmax(axis=1),
                                  CLF2_TEST_PREDICTION.argmax(axis=1))
print CL_REPORT




########################################################################
        # calculate accuracy and hamming loss
########################################################################
ACCURACY = accuracy_score(CLF2_TEST_TARGET.values.argmax(axis=1),
                          CLF2_TEST_PREDICTION.argmax(axis=1))
HL = hamming_loss(CLF2_TEST_TARGET.values.argmax(axis=1), CLF2_TEST_PREDICTION.argmax(axis=1))
print 'Hamming Loss :', HL
print 'Accuracy :', ACCURACY




########################################################################
        # Save the model weights
        # Change the name if are tweaking parameters
########################################################################
MODEL.save_weights('multiclass_weights.h5')
