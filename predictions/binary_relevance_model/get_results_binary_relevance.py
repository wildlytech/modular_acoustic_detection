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
sys.path.insert(0, "../../")
from youtube_audioset import get_recursive_sound_names, get_all_sound_names
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, WOOD_SOUNDS, \
                             HUMAN_SOUNDS, NATURE_SOUNDS, DOMESTIC_SOUNDS, TOOLS_SOUNDS, BIRD



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
                    action='store', help='Input the path with filename (.pkl)')
RESULT = PARSER.parse_args()



##############################################################################
            # Get all sound names
##############################################################################
AMBIENT_SOUNDS, IMPACT_SOUNDS = get_all_sound_names()
explosion_sounds = get_recursive_sound_names(EXPLOSION_SOUNDS)
motor_sounds = get_recursive_sound_names(MOTOR_SOUNDS)
wood_sounds = get_recursive_sound_names(WOOD_SOUNDS)
human_sounds = get_recursive_sound_names(HUMAN_SOUNDS)
nature_sounds = get_recursive_sound_names(NATURE_SOUNDS)
domestic_sounds = get_recursive_sound_names(DOMESTIC_SOUNDS)
tools = get_recursive_sound_names(TOOLS_SOUNDS)
bird = get_recursive_sound_names(BIRD)
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
                   'Domestic_animals',
                   'Tools']


ALL_SOUND_LIST = explosion_sounds + motor_sounds + human_sounds + \
                 nature_sounds + domestic_sounds + tools



##############################################################################
          # Map all the sounds into their respective classes
          # comment any of the labels that are not in dataset
##############################################################################
DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Motor_sound' if x  in motor_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Explosion_sound' if x  in explosion_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Nature_sound' if x  in nature_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Human_sound' if x  in human_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Domestic_animals' if x  in domestic_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Tools' if x  in tools else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Wood_sound' if x  in wood_sounds else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Bird_sound' if x in bird else x for x in arr])
# DATA_FRAME['labels_new']=DATA_FRAME['labels_new'].apply(lambda arr: ['Wild_animals' if x  in Wild_animals else x for x in arr])


##############################################################################
            # print out the shape and sample data
##############################################################################
print DATA_FRAME.head()
print DATA_FRAME.shape


##############################################################################
          # Binarize the labels. Its a Multi-label binarizer
##############################################################################
NAME_BIN = MultiLabelBinarizer().fit(DATA_FRAME['labels_new'])
NEW_LABELS_BINARIZED = NAME_BIN.transform(DATA_FRAME['labels_new'])
NEW_LABELS_BINARIZED_ALL = pd.DataFrame(NEW_LABELS_BINARIZED, columns=NAME_BIN.classes_)
NEW_LABELS_BINARIZED = NEW_LABELS_BINARIZED_ALL[ALL_SOUND_NAMES]



###############################################################################
      # UnComment any of the labels which are not in the dataset
##############################################################################
# NEW_LABELS_BINARIZED["Motor_sound"] = [0] * DATA_FRAME.shape[0]
# NEW_LABELS_BINARIZED["Domestic_animals"] = [0] * DATA_FRAME.shape[0]
# NEW_LABELS_BINARIZED["Tools"] = [0] * DATA_FRAME.shape[0]
# NEW_LABELS_BINARIZED["Nature_sound"] = [0] * DATA_FRAME.shape[0]
# NEW_LABELS_BINARIZED["Human_sound"] = [0] * DATA_FRAME.shape[0]
# LABELS_BINARIZED["Bird_sound"] = [0] * DATA_FRAME.shape[0]
# NEW_LABELS_BINARIZED["Explosion_sound"] = [0] * DATA_FRAME.shape[0]



##############################################################################
      # Create a datafame with only which are available
##############################################################################
LABELS_BINARIZED = pd.DataFrame()
LABELS_BINARIZED["Domestic"] = NEW_LABELS_BINARIZED['Domestic_animals']
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
CLF2_TRAIN_TARGET = LABELS_FILTERED


##############################################################################
      # Implementing using the keras usual training techinque
##############################################################################
MODEL = create_keras_model()

CLF2_TRAIN_PREDICTION = []
CLF2_TRAIN_PREDICTION_PROB = []

for each_embedding in DF_TRAIN["features"].values.tolist():
    prediction_list = []
    prediction_rounded = []
    for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
        pred_prob, pred_round = generate_before_predict_BR.main(True, 1, each_embedding, each_model)
        pred_prob = pred_prob[0][0] * 100
        pred_round = pred_round[0][0]
        prediction_list.append("{0:.2f}".format(pred_prob))
        prediction_rounded.append(pred_round)
    CLF2_TRAIN_PREDICTION.append(prediction_list)
    CLF2_TRAIN_PREDICTION_PROB.append(prediction_rounded)



##############################################################################
      # Predict on train and test data
##############################################################################
# CLF2_TRAIN_PREDICTION  = np.array([[0] if value<=0.40 else [1] for value in CLF2_TRAIN_PREDICTION_PROB])




##############################################################################
          # To get the Misclassified examples
##############################################################################
DF_TRAIN['actual_labels'] = np.split(CLF2_TRAIN_TARGET.values, DF_TRAIN.shape[0])
DF_TRAIN['predicted_labels'] = np.split(CLF2_TRAIN_PREDICTION, DF_TRAIN.shape[0])
DF_TRAIN['predicted_prob'] = np.split(CLF2_TRAIN_PREDICTION_PROB, DF_TRAIN.shape[0])
MISCLASSIFED_ARRAY = CLF2_TRAIN_PREDICTION != CLF2_TRAIN_TARGET
MISCLASSIFIED_EXAMPLES = np.any(MISCLASSIFED_ARRAY, axis=1)
print 'Misclassified number of examples :', DF_TRAIN[MISCLASSIFIED_EXAMPLES].shape[0]



##############################################################################
          # uncomment if misclassified examples are to be saved
##############################################################################
if RESULT.save_misclassified_examples:
    with open("misclassified_examples_br_model.pkl", "w") as f:
        pickle.dump(DF_TRAIN[MISCLASSIFIED_EXAMPLES][["wav_file",
                                                      "actual_labels",
                                                      "predicted_labels",
                                                      "predicted_prob"]], f)
else:
    pass



##############################################################################
          # Print confusion matrix and classification_report
##############################################################################
print CLF2_TRAIN_TARGET.values.argmax(axis=1).shape
print '        Confusion Matrix          '
print '============================================'
RESULT = confusion_matrix(CLF2_TRAIN_TARGET.values.argmax(axis=1),
                          CLF2_TRAIN_PREDICTION.argmax(axis=1))
print RESULT



##############################################################################
        # print classification report
##############################################################################
print '                 Classification Report      '
print '============================================'
CL_REPORT = classification_report(CLF2_TRAIN_TARGET.values.argmax(axis=1),
                                  CLF2_TRAIN_PREDICTION.argmax(axis=1))
print CL_REPORT



##############################################################################
        # calculate accuracy and hamming loss
##############################################################################
ACCURACY = accuracy_score(CLF2_TRAIN_TARGET.values.argmax(axis=1),
                          CLF2_TRAIN_PREDICTION.argmax(axis=1))
HL = hamming_loss(CLF2_TRAIN_TARGET.values.argmax(axis=1), CLF2_TRAIN_PREDICTION.argmax(axis=1))
print 'Hamming Loss :', HL
print 'Accuracy :', ACCURACY



##############################################################################
        # save the prediction in pickle format
##############################################################################
with open("predictions.pkl", "w") as file_obj:
    pickle.dump(DF_TRAIN, file_obj)
