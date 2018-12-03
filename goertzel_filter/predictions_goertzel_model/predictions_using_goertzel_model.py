"""
predictions are done on the audiomoth files
"""
import argparse
import keras
from keras.models import Sequential, Model
from keras.layers import Input
import predictions_on_weights
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D, TimeDistributed, MaxPooling2D
import audiomoth_function_for_goertzel_model as aud_goertzel


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

#Define the goertzel model same as model from which the weights are saved
INPUTS = Input(shape=(10, 8000, 4))
CONV_1 = TimeDistributed(Conv1D(100,
	                               kernel_size=200,
	                               strides=100,
	                               activation='relu',
	                               padding='same'),
                         input_shape=(10, 8000, 4))(INPUTS)
CONV_2 = TimeDistributed(Conv1D(100,
	                               kernel_size=4,
	                               activation='relu',
	                               padding='same'))(CONV_1)
MAX_POOL = TimeDistributed(MaxPooling1D(80))(CONV_2)
DENSE_1 = TimeDistributed(Dense(60, activation='relu'))(MAX_POOL)
DENSE_2 = TimeDistributed(Dense(50, activation='relu'))(DENSE_1)
DENSE_3 = TimeDistributed(Dense(1, activation='sigmoid'))(DENSE_2)
MAX_POOL_2 = MaxPooling2D((10, 1))(DENSE_3)
PREDICTIONS = Flatten()(MAX_POOL_2)
MODEL = Model(inputs=[INPUTS], outputs=[PREDICTIONS])
print MODEL.summary()

# Load the saved weights and predict on the audiomoth  recordings
MODEL.load_weights('../Goertzel_model_8k_weights_time.h5')

#call the audiomoth function to predict on the frequency components of audio moth WAV files
TEST_VALUES, DATAFRAME = aud_goertzel.dataframe_with_frequency_components(RESULT.audio_files_path,
                                                                          RESULT.path_for_goertzel_components)
PREDICTIONS = MODEL.predict(TEST_VALUES).ravel()
DATAFRAME['predictions_prob'] = PREDICTIONS
DATAFRAME['predictions'] = PREDICTIONS.ravel().round()

#save it in a CSV file
DATAFRAME[['wav_file',
		         'Label_1',
		         'Label_2',
		         'Label_3',
		         'predictions_prob',
		         'predictions']].to_csv('predictions_by_goertzel_model.csv')
