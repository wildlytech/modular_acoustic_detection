import pandas as pd
import keras
import audiomoth_function
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop, Adam
import numpy as np
import argparse

# parsing the inputs given  
parser= argparse.ArgumentParser(description = 'Input the path of the audio files recorded from audiomoth ( .WAV files ) and thier embeddings  ')
parser.add_argument('-path_to_audio_files','--path_to_audio_files', action ='store' , help = 'Input the path')
parser.add_argument('-path_to_embeddings','--path_to_embeddings', action ='store' , help = 'Input the path')
result = parser.parse_args()


#Define the model
def create_keras_model():
    # create model
    model = Sequential()
    model.add(Conv1D(500, input_shape=(1280,1), kernel_size=128, strides=128, activation='relu', padding='same'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(MaxPooling1D(10))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4,epsilon=1e-8), metrics=['accuracy'])
    return model

# call the model and load the weights
model = create_keras_model()
model.load_weights('multiclass_weights.h5')


# call the audiomoth function to get the dataframe with embeddings and then filter out audio files that are not having [ 10*128 ] embeddings size
df = audiomoth_function.embeddings_on_dataframe(result.path_to_audio_files,result.path_to_embeddings)
df_filtered = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
print df_filtered.head()

#reshape the data according to model's input 
X_test = np.array(df_filtered.features.apply(lambda x: x.flatten()).tolist())
test_data = X_test.reshape((-1,1280,1))

#predict the data using the loaded model 
predictions= model.predict(test_data).ravel().round()
df['predicted_labels']=np.split(predictions,df.shape[0])

#Save it to  csv file to see the results
df.to_csv('audiomoth_prediction.csv')
