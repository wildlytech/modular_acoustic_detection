import keras
from keras.models import Sequential, Model
from keras.layers import Input
import pickle
import predictions_on_weights
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D, TimeDistributed, MaxPooling2D
import audiomoth_function


#give the path where the frequency components are saved ( '.pkl' files )
frequency_component_files = '/media/wildly/1TB-HDD/goertzel_data_ervikulam_8k/'
audio_files_path = '/media/wildly/1TB-HDD/AudioMoth/test_wave/'

#Define the goertzel model same as model from which the weights are saved
inputs = Input(shape=(10,8000,4))
conv_1= TimeDistributed(Conv1D(100, kernel_size=200,strides=100,activation='relu',padding='same'),input_shape=(10,8000,4))(inputs)
conv_2 = TimeDistributed(Conv1D(100, kernel_size=4,activation='relu',padding='same'))(conv_1)
max_pool = TimeDistributed(MaxPooling1D(80))(conv_2)
dense_1 = TimeDistributed(Dense(60, activation = 'relu'))(max_pool)
dense_2 = TimeDistributed(Dense(50, activation = 'relu'))(dense_1)
dense_3 = TimeDistributed(Dense(1, activation='sigmoid'))(dense_2)
max_pool_2 = MaxPooling2D((10,1))(dense_3)
predictions = Flatten()(max_pool_2)
model = Model(inputs=[inputs],outputs=[predictions])
print model.summary()

#Load the saved weights and predict on the audiomoth  recordings
model.load_weights('../Goertzel_model_8k_weights_time.h5')

#call the audiomoth function to predict on the frequency components of audio moth WAV files
test_values, df = audiomoth_function_for_goertzel_model.dataframe_with_frequency_components(audio_files_path,frequency_component_files)
predictions = model.predict(test_values).ravel()
df['predictions_prob'] = predictions
df['predictions'] = predictions.ravel().round()

#save it in a CSV file
df[['wav_file', 'Label_1','Label_2', 'Label_3', 'predictions_prob', 'predictions']].to_csv('predictions_by_goertzel_model.csv')
