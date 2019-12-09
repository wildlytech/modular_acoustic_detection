"""
Testing a BR - Model
"""
import pickle
import argparse
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, AtrousConvolution1D
from keras.optimizers import Adam
import generate_before_predict_BR
import model_function_binary_relevance


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
PARSER.add_argument('-results_in_csv', '--results_in_csv',
                    action='store', help='Input the filename (.csv)')

RESULT = PARSER.parse_args()



##############################################################################
        # read the dataframe with feature and labels_name column
##############################################################################
with open(RESULT.path_for_dataframe_with_features, "rb") as f:
    DATA_FRAME = pickle.load(f)
DATA_FRAME.index = range(0, DATA_FRAME.shape[0])




##############################################################################
        # Filtering the sounds that are exactly 10 seconds
##############################################################################
DF_TRAIN = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]



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



##############################################################################
      # Implementing using the keras usual training techinque
##############################################################################
MODEL = create_keras_model()


for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
    prediciton_prob, prediction = model_function_binary_relevance.predictions_batch_wavfiles(CLF2_TRAIN, each_model)
    DF_TRAIN[each_model+"_Probability"] = prediciton_prob
    DF_TRAIN[each_model+"_Prediction"] = prediction




##############################################################################
      # Implementing using the keras usual training techinque
##############################################################################
if RESULT.results_in_csv:
    DF_TRAIN.drop(["features"], axis=1).to_csv(RESULT.results_in_csv)
else:
    print DF_TRAIN.head()
