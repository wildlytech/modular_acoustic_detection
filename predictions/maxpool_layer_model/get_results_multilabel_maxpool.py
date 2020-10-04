"""
Traning a Mulit-label Model
"""
#Import the necessary functions and libraries
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
sys.path.insert(0, "../../")
from youtube_audioset import get_recursive_sound_names, get_all_sound_names
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, WOOD_SOUNDS, \
                             HUMAN_SOUNDS, NATURE_SOUNDS, DOMESTIC_SOUNDS, TOOLS_SOUNDS




##################################################################################
              # Description and Help
##################################################################################
DESCRIPTION = 'Gets the predictions of sounds. \n \
               Input base dataframe file path \
               with feature (Embeddings) and labels_name column in it.'

HELP = 'Input the path for dataframe(.pkl format) with features and labels_name'



##############################################################################
          # Parsing the inputs given
##############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_for_dataframe_with_FL', '--path_for_dataframe_with_FL',
                    action='store', help=HELP)
PARSER.add_argument('-path_for_maxpool_saved_weights_file', '--path_for_maxpool_saved_weights_file',
                    action='store', help='Input the path (.h5) File')
PARSER.add_argument('-csvfilename_to_save_predictions', '--csvfilename_to_save_predictions',
                    action='store', help='Input the filename to save in (.csv) format')
PARSER.add_argument('-path_to_save_prediciton_dataframe', '--path_to_save_prediciton_dataframe',
                    action='store', help='Input the filename to save in (.pkl) format')

RESULT = PARSER.parse_args()






##################################################################################
            # Get all sound names
##################################################################################
explosion_sounds = get_recursive_sound_names(EXPLOSION_SOUNDS, "../../")
motor_sounds = get_recursive_sound_names(MOTOR_SOUNDS, "../../")
wood_sounds = get_recursive_sound_names(WOOD_SOUNDS, "../../")
human_sounds = get_recursive_sound_names(HUMAN_SOUNDS, "../../")
nature_sounds = get_recursive_sound_names(NATURE_SOUNDS, "../../")
domestic_sounds = get_recursive_sound_names(DOMESTIC_SOUNDS, "../../")
tools = get_recursive_sound_names(TOOLS_SOUNDS, "../../")
# bird = get_recursive_sound_names(BIRD)
#wild_animals=get_recursive_sound_names(Wild_animals)



##################################################################################
            # Reading the dataframe
##################################################################################
with open(RESULT.path_for_dataframe_with_FL, 'rb') as file_obj:
    DATA_FRAME = pickle.load(file_obj)
DATA_FRAME.index = list(range(0, DATA_FRAME.shape[0]))



##################################################################################
                  # Different classes of sounds.
    # You can increase the class by adding the necesssary sounds of that class
##################################################################################
ALL_SOUND_NAMES = ['Motor_sound',
                   'Explosion_sound',
                   'Human_sound',
                   'Nature_sound',
                   'Domestic_animals',
                   'Tools']
                   # "Bird_sound"]

##################################################################################
      # Automatically get the labels that are present in data
##################################################################################
def get_labels_present(dataframe):
    columns = dataframe.columns
    present_columns = []
    for each_column in ALL_SOUND_NAMES:
        if each_column in columns:
            present_columns.append(each_column)
        else:
            pass
    return present_columns

##################################################################################
      # Get the labels that are not present in the data
##################################################################################
def get_remaining_columns(present_columns):
    absence_column = []
    for each_column in ALL_SOUND_NAMES:
        if not each_column in present_columns:
            absence_column.append(each_column)
        else:
            pass
    return absence_column



##################################################################################

##################################################################################
ALL_SOUND_LIST = explosion_sounds + motor_sounds + human_sounds + \
                 nature_sounds + domestic_sounds + tools



##################################################################################
            # Map all the sounds into their respective classes
##################################################################################
DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Motor_sound' if x  in motor_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Explosion_sound' if x  in explosion_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Nature_sound' if x  in nature_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Human_sound' if x  in human_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Domestic_animals' if x  in domestic_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Tools' if x  in tools else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Wood_sound' if x  in wood_sounds else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Bird_sound' if x in bird else x for x in arr])
# DATA_FRAME['labels_new']=DATA_FRAME['labels_new'].apply(lambda arr: ['Wild_animals' if x  in Wild_animals else x for x in arr])



##################################################################################
          # Binarize the labels. Its a Multi-label binarizer
##################################################################################
NAME_BIN = MultiLabelBinarizer().fit(DATA_FRAME['labels_new'])
LABELS_BINARIZED = NAME_BIN.transform(DATA_FRAME['labels_new'])
LABELS_BINARIZED_ALL = pd.DataFrame(LABELS_BINARIZED, columns=NAME_BIN.classes_)
COLUMNS_PRESENT = get_labels_present(LABELS_BINARIZED_ALL)
if COLUMNS_PRESENT:
    LABELS_BINARIZED = LABELS_BINARIZED_ALL[COLUMNS_PRESENT]
else:
    print("No Relevant Labels Found")
    sys.exit(1)



##################################################################################
          # Uncomment if label is not present in dataset
##################################################################################
ABSENT_COLUMNS = get_remaining_columns(COLUMNS_PRESENT)
if ABSENT_COLUMNS:
    for each_col in ABSENT_COLUMNS:
        LABELS_BINARIZED[each_col] = [0] * DATA_FRAME.shape[0]
else:
    pass


##################################################################################
          # Arrange them in the order same as training
##################################################################################
LABELS_BINARIZED = LABELS_BINARIZED[ALL_SOUND_NAMES]
print(LABELS_BINARIZED.mean())




##################################################################################
          # Filtering the sounds that are exactly 10 seconds
##################################################################################
DF_TRAIN = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
LABELS_FILTERED = LABELS_BINARIZED.loc[DF_TRAIN.index, :]



##################################################################################
          # preprecess the data into required structure
##################################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255



##################################################################################
          # create the keras model
##################################################################################
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
    model.add(Dense(6, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model



##################################################################################
          # reshaping the train and test data so as to align with input for model
##################################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET = LABELS_FILTERED




##################################################################################
          # Implementing using the keras usual training techinque
##################################################################################
MODEL = create_keras_model()

if RESULT.path_for_maxpool_saved_weights_file:
    MODEL.load_weights(RESULT.path_for_maxpool_saved_weights_file)
else:
    MODEL.load_weights("multi_class_weights_waynad_nandi_2_nandi_1(4k-m_6k-h_4K-n).h5")




##################################################################################
          # Predict on train and test data
##################################################################################
CLF2_TRAIN_PREDICTION = MODEL.predict(CLF2_TRAIN).round()
CLF2_TRAIN_PREDICTION_PROB = MODEL.predict(CLF2_TRAIN)
# CLF2_TRAIN_PREDICTION  = [0 if value<=0.45 else 1 for value in CLF2_TRAIN_PREDICTION_PROB]





##################################################################################
              # To get the Misclassified examples
##################################################################################
DF_TRAIN['actual_labels'] = np.split(CLF2_TRAIN_TARGET.values, DF_TRAIN.shape[0])
DF_TRAIN['predicted_labels'] = np.split(CLF2_TRAIN_PREDICTION, DF_TRAIN.shape[0])
DF_TRAIN['predicted_prob'] = np.split(CLF2_TRAIN_PREDICTION_PROB, DF_TRAIN.shape[0])
MISCLASSIFED_ARRAY = CLF2_TRAIN_PREDICTION != CLF2_TRAIN_TARGET
MISCLASSIFIED_EXAMPLES = np.any(MISCLASSIFED_ARRAY, axis=1)




# with open("misclassified_weights_Birds_added_maxpool_at_end_4times_500units.pkl","w") as f:
#   pickle.dump(DF_TRAIN[MISCLASSIFIED_EXAMPLES][["wav_file","actual_labels","predicted_labels","predicted_prob"]],f)



##################################################################################
          # Print confusion matrix
##################################################################################
print(CLF2_TRAIN_TARGET.values.argmax(axis=1).shape)
print('        Confusion Matrix          ')
print('============================================')
RESULT_ = confusion_matrix(CLF2_TRAIN_TARGET.values.argmax(axis=1),
                           CLF2_TRAIN_PREDICTION.argmax(axis=1))
print(RESULT_)



##################################################################################
          # print classification report
##################################################################################
print('                 Classification Report      ')
print('============================================')
CL_REPORT = classification_report(CLF2_TRAIN_TARGET.values.argmax(axis=1),
                                  CLF2_TRAIN_PREDICTION.argmax(axis=1))
print(CL_REPORT)



##################################################################################
          # calculate accuracy and hamming loss
##################################################################################
ACCURACY = accuracy_score(CLF2_TRAIN_TARGET.values.argmax(axis=1),
                          CLF2_TRAIN_PREDICTION.argmax(axis=1))
HL = hamming_loss(CLF2_TRAIN_TARGET.values.argmax(axis=1), CLF2_TRAIN_PREDICTION.argmax(axis=1))

print('Hamming Loss :', HL)
print('Accuracy :', ACCURACY)


##################################################################################
          # path to save the predictions in pkl format
##################################################################################
if RESULT.path_to_save_prediciton_dataframe:
    with open(RESULT.path_to_save_prediciton_dataframe, "w") as file_obj:
        pickle.dump(DF_TRAIN, file_obj)
else:
    print("Predictions not saved in DataFrame format (.pkl). \
You won't be able to see Multilabel predictions metrics \
To save, pass the appropiate command line argument.")



##################################################################################
          # path to save predicitons in csv file
##################################################################################
if RESULT.csvfilename_to_save_predictions:
    DF_TRAIN.drop(["features"], axis=1).to_csv(RESULT.csvfilename_to_save_predictions)
else:
    print("Predictions not saved in csv file. \
To save pass the appropiate command line argument.")

