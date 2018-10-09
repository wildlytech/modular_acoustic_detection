import pandas as pd
import keras
import audiomoth_function
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop, Adam
import numpy as np


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


# call the audiomoth function to get the test data for predictions
test_data , df = audiomoth_function.embeddings_on_dataframe()
predictions= model.predict(test_data).ravel().round()
df['predicted_labels']=np.split(predictions,df.shape[0])

#Save it to  csv file to see the results
df[['wav_file','predicted_labels','Label_1','Label_2','Label_3']].to_csv('GKVK_prediction.csv')
