"""
predictions are done on the audiomoth files
"""
import sys
import pickle
import argparse
import numpy as np
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed, MaxPooling2D



###############################################################################
          # Description and Help
###############################################################################
DESCRIPTION = "Compares the prediction my goertzel model and annotated labels"
HELP_AUDIO = "Path for Dataframe with features( .pkl )"
HELP_GOERTZEL = "Path to write predictions in csv( .csv )"



###############################################################################
          # parse the input arguments given from command line
###############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_dataframe_with_features',
                    '--path_dataframe_with_features', action='store',
                    help=HELP_AUDIO)
PARSER.add_argument('-path_for_saved_weights',
                    '--path_for_saved_weights', action='store',
                    help='Weights file (.h5)')
PARSER.add_argument("-path_to_write_prediction_csv", action="store",
                    help=HELP_GOERTZEL)
RESULT = PARSER.parse_args()




###############################################################################
    # Define the goertzel model same as model from which the weights are saved
###############################################################################
INPUTS = Input(shape=(10, 8000, 4))
MAXPOOL_1 = TimeDistributed(MaxPooling1D(80))(INPUTS)
CONV_1 = TimeDistributed(Conv1D(20,
                                kernel_size=100,
                                strides=50,
                                activation='relu',
                                padding='same'))(MAXPOOL_1)
MAXPOOL_2 = TimeDistributed(MaxPooling1D(2))(CONV_1)
DENSE_1 = TimeDistributed(Dense(50, activation='relu'))(MAXPOOL_2)
DENSE_2 = TimeDistributed(Dense(50, activation='relu'))(DENSE_1)
DENSE_3 = TimeDistributed(Dense(50, activation='relu'))(DENSE_2)
DENSE_4 = TimeDistributed(Dense(50, activation='relu'))(DENSE_3)
DENSE_5 = TimeDistributed(Dense(1, activation='sigmoid'))(DENSE_4)
MAX_POOL_3 = MaxPooling2D((10, 1))(DENSE_5)
PREDICTIONS = Flatten()(MAX_POOL_3)
MODEL = Model(inputs=[INPUTS], outputs=[PREDICTIONS])
print(MODEL.summary())



###############################################################################
      # Load the saved weights and predict on the audiomoth  recordings
###############################################################################

if RESULT.path_for_saved_weights:
    MODEL.load_weights(RESULT.path_for_saved_weights)
else:
    print("No weights File given.")
    sys.exit(1)

###############################################################################
      # Read the dataframe with columns features and labels name
###############################################################################
if RESULT.path_dataframe_with_features:
    with open(RESULT.path_dataframe_with_features, "rb") as file_obj:
        DF_TEST = pickle.load(file_obj)
else:
    print("Dataframe not Found.")
    sys.exit(1)


###############################################################################
            # Read all the test data first
###############################################################################
CLF1_TEST = []
print('Reading Test files ..!')

for each_emb, each_wav in zip(DF_TEST['features'].tolist(), DF_TEST["wav_file"].tolist()):
    # Read all the files that are splitted as test in the path directory specified
    try:
        CLF1_TEST.append(np.dstack((each_emb[0].reshape((10, 8000)),
                                    each_emb[1].reshape((10, 8000)),
                                    each_emb[2].reshape((10, 8000)),
                                    each_emb[3].reshape((10, 8000)))))

    #except any error then remove that file manually and then run the process again
    except OSError:
        print('Test Pickling Error: ', each_wav)


###############################################################################
        # Preprocess the data as per the input of the model
###############################################################################
CLF1_TEST = np.array(CLF1_TEST).reshape((-1, 10, 8000, 4))
CLF1_TEST = CLF1_TEST / np.linalg.norm(CLF1_TEST)


###############################################################################
        # Run the predictions on the data
###############################################################################
PREDICTIONS = MODEL.predict(CLF1_TEST).ravel()
DF_TEST['predictions_prob'] = PREDICTIONS
DF_TEST['predictions'] = PREDICTIONS.ravel().round()


###############################################################################
          # save it in a CSV file
###############################################################################
if RESULT.path_to_write_prediction_csv:
    DF_TEST.drop(["features"], axis=1).to_csv(RESULT.path_to_write_prediction_csv)
else:
    print("Predictions are not saved. Give appropiate argument to save the predictions")
