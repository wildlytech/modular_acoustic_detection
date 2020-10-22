"""
Predictions are returned for a single wav file
"""
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.optimizers import Adam



##############################################################################

##############################################################################
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
    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    # print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model



##############################################################################

##############################################################################
def predictions_wavfile(data, model_type):
    """
    Takes the embeddings of wav file and runs the predictions on it
    """
    # call the model and load the weights
    model = create_keras_model()

    if model_type == "Motor":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Motor_BR_Model/4_Layer_variants/binary_relevance_motor_realised_multilabel_weights_mixedsounds_MH_MN_ME_added_maxpool_at_end_4times_500units_sigmoid.h5")
    elif model_type == "Human":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Human_BR_Model/4_Layer_Variant/binary_relevance_human_multilabel_weights_mixedsounds_HM_HN_HT_added_maxpool_at_end_4times_500units_sigmoid.h5")
    elif model_type == "Explosion":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Explosion_BR_Model/4_Layer_Variant/binary_relevance_explosion_realised_multilabel_weights_mixedsounds_EM_EN_EH_added_maxpool_at_end_4times_500units_sigmoid.h5")
    elif model_type == "Tools":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Tools_BR_Model/4_Layer_Variants/binary_relevance_tools_realised_multilabel_weights_mixedsounds_TM_TD_TH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    elif model_type == "Domestic":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Domestic_BR_Model/binary_relevance_domestic_realised_multilabel_weights_mixedsounds_DM_DT_DH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    elif model_type == "Nature":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Nature_BR_Model/binary_relevance_nature_realised_multilabel_weights_mixedsounds_NM_NE_NH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    else:
        print("Invalid Model Selected")

    if data.shape[0] == 10:
        test_data = data.reshape((-1, 1280, 1))
        if len(test_data) == 1:
            predictions_prob = model.predict(test_data)
            predictions = predictions_prob.round()
        else:
            print("testing", len(test_data))
            # predict the data using the loaded model
            predictions_prob = model.predict(test_data).ravel()
            predictions = predictions_prob.round()
        return predictions_prob, predictions
    else:
        return None, None



##############################################################################

##############################################################################

def predictions_batch_wavfiles(data, model_type):
    """
    Takes the embeddings of wav file and runs the predictions on it
    """
    # call the model and load the weights
    model = create_keras_model()

    if model_type == "Motor":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Motor_BR_Model/4_Layer_variants/binary_relevance_motor_realised_multilabel_weights_mixedsounds_MH_MN_ME_added_maxpool_at_end_4times_500units_sigmoid.h5")
    elif model_type == "Human":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Human_BR_Model/4_Layer_Variant/binary_relevance_human_multilabel_weights_mixedsounds_HM_HN_HT_added_maxpool_at_end_4times_500units_sigmoid.h5")
    elif model_type == "Explosion":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Explosion_BR_Model/4_Layer_Variant/binary_relevance_explosion_realised_multilabel_weights_mixedsounds_EM_EN_EH_added_maxpool_at_end_4times_500units_sigmoid.h5")
    elif model_type == "Tools":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Tools_BR_Model/4_Layer_Variants/binary_relevance_tools_realised_multilabel_weights_mixedsounds_TM_TD_TH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    elif model_type == "Domestic":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Domestic_BR_Model/binary_relevance_domestic_realised_multilabel_weights_mixedsounds_DM_DT_DH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    elif model_type == "Nature":
        model.load_weights("predictions/binary_relevance_model/Binary_Relevance_Models/Nature_BR_Model/binary_relevance_nature_realised_multilabel_weights_mixedsounds_NM_NE_NH_added_maxpool_at_end_4times_500units_sigmoid_Test.h5")
    else:
        print("Invalid Model Selected")


    predictions_prob = model.predict(data)
    predictions = predictions_prob.round()
    return predictions_prob, predictions

