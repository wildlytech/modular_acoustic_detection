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
PARSER.add_argument('-path_for_wavfile', '--path_for_wavfile',
                    action='store', help='Input the path for 10second wavfile')
PARSER.add_argument('-save_misclassified_examples', '--save_misclassified_examples',
                    action='store', help='Input the path with filename (.pkl)')
RESULT = PARSER.parse_args()



##############################################################################
        # read the dataframe with feature and labels_name column
##############################################################################
EMBEDDINGS = generate_before_predict_BR.main(RESULT.path_for_wavfile, 0, None, None)



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
      # Implementing using the keras usual training techinque
##############################################################################
MODEL = create_keras_model()

CLF2_TRAIN_PREDICTION = []
CLF2_TRAIN_PREDICTION_PROB = []

for each_embedding in [EMBEDDINGS]:
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

##############################################################################
print "Predictions: ", CLF2_TRAIN_PREDICTION
