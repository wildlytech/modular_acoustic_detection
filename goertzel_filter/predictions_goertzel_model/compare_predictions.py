import keras
from keras.models import Sequential, Model
from keras.layers import Input
import pickle
import predicting_without_libraries
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D, TimeDistributed, MaxPooling2D
import audiomoth_function_for_goertzel_model


#give the path where the frequency components are saved ( '.pkl' files )
path_to_goertzel_components = '/media/wildly/1TB-HDD/goertzel_data_ervikulam_8k/'
audio_files_path = '/media/wildly/1TB-HDD/AudioMoth/test_wave/'

inputs = Input(shape=(10,8000,4))
conv_1= TimeDistributed(Conv1D(100, kernel_size=200,strides=100,activation='relu',padding='same'),input_shape=(10,8000,4))(inputs)
conv_2 = TimeDistributed(Conv1D(100, kernel_size=4,activation='relu',padding='same'))(conv_1)
max_pool = TimeDistributed(MaxPooling1D(80))(conv_2)
dense_1 = TimeDistributed(Dense(60, activation = 'relu'))(max_pool)
dense_2 = TimeDistributed(Dense(50, activation = 'relu'))(dense_1)
dense_3 = TimeDistributed(Dense(1, activation='sigmoid'))(dense_2)
max_pool_2 = MaxPooling2D((10,1))(dense_3)
predictions = Flatten()(max_pool_2)
# print conv_1.output
model = Model(inputs=[inputs],outputs=[predictions])
print model.summary()

#Load weights from saved weights file
model.load_weights('Goertzel_model_8k_weights_time.h5')

# get the weights of each layer from loaded model
weights =[]
for layer in model.layers:
	weights.append(layer.get_weights())
print 'Length of  the weights : ', len(weights)

#redict on audiomoth files using keras library
import audiomoth_function_for_goertzel_model
test_values, df = audiomoth_function_for_goertzel_model.dataframe_with_frequency_components(audio_files_path, path_to_goertzel_components)
predictions = model.predict(test_values).ravel()
df['predictions_prob'] = predictions
df['predictions'] = predictions.ravel().round()

#predict on any of the audiomoth file without using any of the keras library
#In this example code we are taking 100th example
ouput = predicting_without_libraries.layer1_predictions( test_values[100], weights )
print df['predictions_prob'][100]
