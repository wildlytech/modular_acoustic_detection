"""
Predictions are returned for a single wav file
"""
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam



############################################################################
		# Create keras model (should be same as saved model archietcture)
############################################################################
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
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model



############################################################################
		# Predicts the ouput for Dense Layer Model
############################################################################
def predictions_wavfile(data):
    """
    Takes the embeddings of wav file and runs the predictions on it
    """
    model = create_keras_model()
    model.load_weights('../../predictions/dense_layer_model/multi_class_weights_waynad_nandi_2_nandi_1(4k-m_6k-h_4K-n).h5')
    test_data = data.reshape((-1, 1280, 1))
    if len(test_data) == 1:
        predictions_prob = model.predict(test_data)
        predictions = predictions_prob.round()
    else:
        #predict the data using the loaded model
        predictions_prob = model.predict(test_data).ravel()
        predictions = predictions_prob.round()
    return predictions_prob, predictions



def predictions_batch_wavfiles(data):
    """
    Takes the embeddings of wav file and runs the predictions on it
    """
    model = create_keras_model()
    model.load_weights('../../predictions/dense_layer_model/multi_class_weights_waynad_nandi_2_nandi_1(4k-m_6k-h_4K-n).h5')
    predictions_prob = model.predict(data).ravel()
    predictions = predictions_prob.round()
    return predictions_prob, predictions