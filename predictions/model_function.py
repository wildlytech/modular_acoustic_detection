"""
Predictions are returned for a single wav file
"""
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

def create_keras_model():
    """
    Create a model
    """
    model = Sequential()
    model.add(Conv1D(500, input_shape=(1280, 1),
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

def predictions_wavfile(data):
    """
    Takes the embeddings of wav file and runs the predictions on it
    """
    # call the model and load the weights
    model = create_keras_model()
    model.load_weights('multiclass_weights.model')
    test_data = data.reshape((-1, 1280, 1))
    #predict the data using the loaded model
    predictions_prob = model.predict(test_data).ravel()
    predictions = predictions_prob.round()
    return predictions_prob, predictions
