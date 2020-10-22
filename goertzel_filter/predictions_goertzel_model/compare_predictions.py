"""
Compares the predictions made using model libaries and without libraries
"""
import argparse
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.layers import Input
from . import predicting_without_libraries
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D, TimeDistributed, MaxPooling2D
from . import audiomoth_function_for_goertzel_model


# Define the constants
DESCRIPTION = "Compares the prediction my goertzel model and annotated labels"
HELP_AUDIO = "Path for audio files ( .WAV )"
HELP_GOERTZEL = "Path for Goertzel freq components ( .pkl )"


#parse the input arguments given from command line
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-audio_files_path',
                    '--audio_files_path', action='store',
                    help=HELP_AUDIO)
PARSER.add_argument('-path_for_goertzel_components',
                    '--path_for_goertzel_components', action='store',
                    help=HELP_GOERTZEL)
RESULT = PARSER.parse_args()


# Define the model architecture using keras functional API
INPUTS = Input(shape=(10, 8000, 4))
CONV_1 = TimeDistributed(Conv1D(100, kernel_size=200, strides=100, activation='relu', padding='same'), input_shape=(10, 8000, 4))(INPUTS)
CONV_2 = TimeDistributed(Conv1D(100, kernel_size=4, activation='relu', padding='same'))(CONV_1)
MAX_POOL = TimeDistributed(MaxPooling1D(80))(CONV_2)
DENSE_1 = TimeDistributed(Dense(60, activation='relu'))(MAX_POOL)
DENSE_2 = TimeDistributed(Dense(50, activation='relu'))(DENSE_1)
DENSE_3 = TimeDistributed(Dense(1, activation='sigmoid'))(DENSE_2)
MAX_POOL_2 = MaxPooling2D((10, 1))(DENSE_3)
PREDICTIONS = Flatten()(MAX_POOL_2)
# print conv_1.output
MODEL = Model(inputs=[INPUTS], outputs=[PREDICTIONS])
print(MODEL.summary())

#Load weights from saved weights file
MODEL.load_weights('Goertzel_model_8k_weights_time.h5')

# get the weights of each layer from loaded model
WEIGHTS = []
for layer in MODEL.layers:
    WEIGHTS.append(layer.get_weights())
print('Length of  the weights : ', len(WEIGHTS))

#redict on audiomoth files using keras library
TEST_VAULES, DATA_FRAME = audiomoth_function_for_goertzel_model.dataframe_with_frequency_components(RESULT.audio_files_path,RESULT.path_to_goertzel_components)
PREDICTIONS_OUT = MODEL.predict(TEST_VAULES).ravel()
DATA_FRAME['predictions_prob'] = PREDICTIONS_OUT
DATA_FRAME['predictions'] = PREDICTIONS_OUT.ravel().round()

#predict on any of the audiomoth file without using any of the keras library
#In this example code we are taking 100th example
OUTPUT = predicting_without_libraries.layer1_predictions(TEST_VAULES[100], WEIGHTS)
print(OUTPUT)
print(DATA_FRAME['predictions_prob'][100])
