"""
Returns a CSV file that has 'predicted_labels' as column
for the audiomoth recorded files 'audiomoth_id'
"""
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import audiomoth_function

# parsing the inputs given
DESCRIPTION = 'Input the path of the audio files recorded from audiomoth \
              ( .WAV files ) and thier embeddings'
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_to_audio_files', '--path_to_audio_files',
                    action='store', help='Input the path')
PARSER.add_argument('-path_to_embeddings', '--path_to_embeddings',
                    action='store', help='Input the path')
RESULT = PARSER.parse_args()

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
    model.add(MaxPooling1D(10))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model

# call the model and load the weights
MODEL = create_keras_model()
MODEL.load_weights('multiclass_weights.h5')

# call the audiomoth function to get the dataframe with embeddings
#and then filter out audio files that are not having [ 10*128 ] embeddings size
DATA_FRAME = audiomoth_function.embeddings_on_dataframe(RESULT.path_to_audio_files,
                                                        RESULT.path_to_embeddings)
DF_FILTERED = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
print DF_FILTERED.head()

#reshape the data according to model's input
X_TEST = np.array(DF_FILTERED.features.apply(lambda x: x.flatten()).tolist())
TEST_DATA = X_TEST.reshape((-1, 1280, 1))

#predict the data using the loaded model
PREDICTIONS = MODEL.predict(TEST_DATA).ravel().round()
DATA_FRAME['predicted_labels'] = np.split(PREDICTIONS, DATA_FRAME.shape[0])

#Save it to  csv file to see the results
DATA_FRAME.drop('features', axis=1).to_csv('audiomoth_prediction.csv')
