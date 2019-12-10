"""
Testing a BR - Model
"""
#Import the necessary functions and libraries
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, AtrousConvolution1D
from keras.optimizers import Adam
import generate_before_predict_BR
import model_function_binary_relevance
sys.path.insert(0, "../../")
from youtube_audioset import get_recursive_sound_names, get_all_sound_names
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, WOOD_SOUNDS, \
                             HUMAN_SOUNDS, NATURE_SOUNDS, DOMESTIC_SOUNDS, TOOLS_SOUNDS



##############################################################################
          # Description and Help
##############################################################################
DESCRIPTION = 'Gets the predictions of sounds. \n \
               Input base dataframe file path \
               with feature (Embeddings) and labels_name column in it.'



##############################################################################
          # Parsing the inputs given
##############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_for_dataframe_with_features', '--path_for_dataframe_with_features',
                    action='store', help='Input the path for dataframe(.pkl format) with features')
PARSER.add_argument('-save_misclassified_examples', '--save_misclassified_examples',
                    action='store', help='Input the directory')
PARSER.add_argument('-path_to_save_prediction_csv', '--path_to_save_prediction_csv',
                    action='store', help='Input the path to save predictions (.csv)')

RESULT = PARSER.parse_args()



##############################################################################
            # Get all sound names
##############################################################################
explosion_sounds = get_recursive_sound_names(EXPLOSION_SOUNDS, "../../")
motor_sounds = get_recursive_sound_names(MOTOR_SOUNDS, "../../")
wood_sounds = get_recursive_sound_names(WOOD_SOUNDS, "../../")
human_sounds = get_recursive_sound_names(HUMAN_SOUNDS, "../../")
nature_sounds = get_recursive_sound_names(NATURE_SOUNDS, "../../")
domestic_sounds = get_recursive_sound_names(DOMESTIC_SOUNDS, "../../")
tools = get_recursive_sound_names(TOOLS_SOUNDS, "../../")
#wild_animals=get_recursive_sound_names(Wild_animals)



##############################################################################
        # read the dataframe with feature and labels_name column
##############################################################################
with open(RESULT.path_for_dataframe_with_features, "rb") as f:
    DATA_FRAME = pickle.load(f)
DATA_FRAME.index = range(0, DATA_FRAME.shape[0])




##############################################################################
      # Comment any of the labels that are not present in the dataset
##############################################################################
ALL_SOUND_NAMES = ['Motor_sound',
                   'Explosion_sound',
                   'Human_sound',
                   'Nature_sound',
                   'Domestic_sound',
                   'Tools_sound']


ALL_SOUND_LIST = explosion_sounds + motor_sounds + human_sounds + \
                 nature_sounds + domestic_sounds + tools


##############################################################################
         # Automatically get the labels that are present in data
##############################################################################
def get_labels_present(dataframe):
    columns = dataframe.columns
    present_columns = []
    for each_column in ALL_SOUND_NAMES:
        if each_column in columns:
            present_columns.append(each_column)
        else:
            pass
    return present_columns



##############################################################################
            # Get the labels that are not present in the data
##############################################################################
def get_remaining_columns(present_columns):
    absence_column = []
    for each_column in ALL_SOUND_NAMES:
        if not each_column in present_columns:
            absence_column.append(each_column)
        else:
            pass
    return absence_column


##############################################################################
          # Map all the sounds into their respective classes
          # comment any of the labels that are not in dataset
##############################################################################
DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Motor_sound' if x  in motor_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Explosion_sound' if x  in explosion_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Nature_sound' if x  in nature_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Human_sound' if x  in human_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Domestic_sounds' if x  in domestic_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Tools_sound' if x  in tools else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Wood_sound' if x  in wood_sounds else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Bird_sound' if x in bird else x for x in arr])
# DATA_FRAME['labels_new']=DATA_FRAME['labels_new'].apply(lambda arr: ['Wild_animals' if x  in Wild_animals else x for x in arr])




##############################################################################
          # Binarize the labels. Its a Multi-label binarizer
##############################################################################
NAME_BIN = MultiLabelBinarizer().fit(DATA_FRAME['labels_new'])
NEW_LABELS_BINARIZED = NAME_BIN.transform(DATA_FRAME['labels_new'])
NEW_LABELS_BINARIZED_ALL = pd.DataFrame(NEW_LABELS_BINARIZED, columns=NAME_BIN.classes_)
COLUMNS_PRESENT = get_labels_present(NEW_LABELS_BINARIZED_ALL)
print COLUMNS_PRESENT
if COLUMNS_PRESENT:
    NEW_LABELS_BINARIZED = NEW_LABELS_BINARIZED_ALL[COLUMNS_PRESENT]
else:
    print "No Relevant Labels Found"
    sys.exit(1)



###############################################################################
      # UnComment any of the labels which are not in the dataset
##############################################################################
ABSENT_COLUMNS = get_remaining_columns(COLUMNS_PRESENT)
if ABSENT_COLUMNS:
    for each_col in ABSENT_COLUMNS:
        NEW_LABELS_BINARIZED[each_col] = [0] * DATA_FRAME.shape[0]
else:
    pass


###############################################################################
     # Arrange all the labels in manner same as trained
###############################################################################
LABELS_BINARIZED = NEW_LABELS_BINARIZED[ALL_SOUND_NAMES]
print LABELS_BINARIZED.mean()




##############################################################################
        # Filtering the sounds that are exactly 10 seconds
##############################################################################
DF_TRAIN = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
LABELS_FILTERED = LABELS_BINARIZED.loc[DF_TRAIN.index, :]



##############################################################################
        # preprecess the data into required structure
##############################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255


##############################################################################
        # create the keras model
##############################################################################
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
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model


##############################################################################
    # reshaping the test data so as to align with input for model
##############################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET_ = LABELS_FILTERED



##############################################################################
      # Implementing using the keras usual training techinque
##############################################################################
MODEL = create_keras_model()

CLF2_TRAIN_PREDICTION_PROB = []



for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
    prediciton_prob, prediction = model_function_binary_relevance.predictions_batch_wavfiles(CLF2_TRAIN, each_model)
    CLF2_TRAIN_PREDICTION = prediction.ravel()
    CLF2_TRAIN_TARGET = LABELS_FILTERED[each_model+"_sound"]
    print CLF2_TRAIN_TARGET.shape
    DF_TRAIN[each_model+"_Probability"] = prediciton_prob.ravel()
    DF_TRAIN[each_model+"_Prediction"] = prediction.ravel()


    ##############################################################################
          # Predict on train and test data
    ##############################################################################
    # CLF2_TRAIN_PREDICTION  = np.array([[0] if value<=0.40 else [1] for value in CLF2_TRAIN_PREDICTION_PROB])




    ##############################################################################
              # To get the Misclassified examples
    ##############################################################################
    DF_TRAIN[each_model+'_actual_labels'] = CLF2_TRAIN_TARGET
    MISCLASSIFED_ARRAY = CLF2_TRAIN_PREDICTION != CLF2_TRAIN_TARGET
    print '\n\nMisclassified number of examples for '+ each_model + " :", DF_TRAIN.loc[MISCLASSIFED_ARRAY].shape[0]


    ##############################################################################
              #  misclassified examples are to be saved
    ##############################################################################
    if RESULT.save_misclassified_examples:
        with open(RESULT.save_misclassified_examples+"misclassified_examples_br_model_"+each_model+".pkl", "w") as f:
            pickle.dump(DF_TRAIN[MISCLASSIFED_ARRAY].drop(["features"], axis=1), f)
    else:
        pass



    ##############################################################################
              # Print confusion matrix and classification_report
    ##############################################################################
    print 'Confusion Matrix for '+ each_model
    print '============================================'
    RESULT_ = confusion_matrix(CLF2_TRAIN_TARGET.values,
                               CLF2_TRAIN_PREDICTION)
    print RESULT_



    ##############################################################################
            # print classification report
    ##############################################################################
    print 'Classification Report for '+ each_model
    print '============================================'
    CL_REPORT = classification_report(CLF2_TRAIN_TARGET.values,
                                      CLF2_TRAIN_PREDICTION)
    print CL_REPORT



    ##############################################################################
            # calculate accuracy and hamming loss
    ##############################################################################
    ACCURACY = accuracy_score(CLF2_TRAIN_TARGET.values,
                              CLF2_TRAIN_PREDICTION)
    HL = hamming_loss(CLF2_TRAIN_TARGET.values, CLF2_TRAIN_PREDICTION)
    print 'Hamming Loss :', HL
    print 'Accuracy :', ACCURACY



##############################################################################
        # save the prediction in pickle format
##############################################################################

if RESULT.path_to_save_prediction_csv:
    DF_TRAIN.drop(["features"], axis=1).to_csv(RESULT.path_to_save_prediction_csv)
else:
    pass
