"""
Returns a CSV file that has 'predicted_labels' as column
for the audiomoth recorded files 'audiomoth_id'
"""
import argparse
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam


##############################################################################
          # Description and Help
##############################################################################
DESCRIPTION = 'Gets the predictions of sounds. \n \
               Input base dataframe file path \
               with feature (Embeddings) column in it.'



##############################################################################
          # Parsing the inputs given
##############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_for_dataframe_with_features', '--path_for_dataframe_with_features',
                    action='store', help='Input the path for dataframe(.pkl format) with features')
PARSER.add_argument('-path_for_maxpool_model_weights', '--path_for_maxpool_model_weights',
                    action='store', help='Input the path (.h5) File')
PARSER.add_argument('-csvfilename_to_save_predictions', '--csvfilename_to_save_predictions',
                    action='store', help='Input the filename to save in (.csv) format')
RESULT = PARSER.parse_args()



##############################################################################
          # Read the dataframe
##############################################################################
with open(RESULT.path_for_dataframe_with_features, 'rb') as file_obj:
    DATA_FRAME = pickle.load(file_obj)




##############################################################################
          # Create Keras model architecture
##############################################################################
def create_keras_model():
    """
    create model
    """
    model = Sequential()
    model.add(Conv1D(500,
                     input_shape=(1280, 1),
                     kernel_size=128,
                     strides=128,
                     activation='relu',
                     padding='same'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model



##############################################################################
        # call the model and load the weights
##############################################################################
MODEL = create_keras_model()
if RESULT.path_for_maxpool_model_weights:
    MODEL.load_weights(RESULT.path_for_maxpool_model_weights)
else:
    MODEL.load_weights("multi_class_weights_waynad_nandi_2_nandi_1(4k-m_6k-h_4K-n).h5")



##############################################################################
        # Filter out sounds which are not equal to 10 seconds
##############################################################################
DF_FILTERED = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
DF_FILTERED.index = list(range(0, DF_FILTERED.shape[0]))


##############################################################################
        # reshape the data according to model's input
##############################################################################
X_TEST = np.array(DF_FILTERED.features.apply(lambda x: x.flatten()).tolist())
TEST_DATA = X_TEST.reshape((-1, 1280, 1))



##############################################################################
        # predict the data using the loaded model
##############################################################################
PREDICTIONS = MODEL.predict(TEST_DATA).ravel().round()
PREDICTIONS_PROB = MODEL.predict(TEST_DATA).ravel()
for index, each_label in enumerate(["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]):
    DF_FILTERED[each_label+"_Prediction"] = np.array(np.split(PREDICTIONS, DF_FILTERED.shape[0]))[:, index]
    DF_FILTERED[each_label+"_Probability"] = np.array(np.split(PREDICTIONS_PROB, DF_FILTERED.shape[0]))[:, index]


##############################################################################
        # Save it to  csv file to see the results
##############################################################################
DF_FILTERED.drop(['features'], axis=1).to_csv(RESULT.csvfilename_to_save_predictions)
