"""
Testing a BR - Model
"""
import argparse
import pandas as pd
import numpy as np
from tensorflow.compat.v1.keras import backend as K
from . import generate_before_predict_BR, get_results_binary_relevance


def predict_on_embedding(embedding, label_names, config_datas):
    '''
    Predict on single embedding for audio clip
    '''
    prediction_probs = []
    prediction_rounded = []

    for label_name in label_names:

        config_data = config_datas[label_name]

        # Clear keras session before all predictions with model
        K.clear_session()

        model = get_results_binary_relevance.load_model(config_data["networkCfgJson"],
                                                        config_data["train"]["outputWeightFile"])

        #######################################################################
        # Predict on test data
        #######################################################################

        if embedding.shape[0] < 10:
            # Pad by repeating the last second in embedding till we get to 10 seconds
            padding = np.ones((10-embedding.shape[0], embedding.shape[1]))*embedding[-1,:]
            embedding = np.vstack([embedding, padding])

        # If the clip is longer then 10 seconds, then predict on multiple 10-second
        # windows within the clip. Take the max probability across all windows.
        pred_prob = 0
        for index in np.arange(10,embedding.shape[0]+1):
            windowed_embedding = embedding[(index-10):index,:].reshape((1,10*128,1))
            pred_prob = max(pred_prob, model.predict(windowed_embedding).ravel()[0])

        # Clear keras session after all predictions with model
        K.clear_session()

        pred_round = np.round(pred_prob)
        pred_prob = pred_prob * 100

        prediction_probs.append(pred_prob)
        prediction_rounded.append(pred_round)

    return prediction_probs, prediction_rounded


def main(predictions_cfg_json, path_for_wavfile):

    ###########################################################################
    # read the dataframe with feature and labels_name column
    ###########################################################################
    EMBEDDINGS = generate_before_predict_BR.main(path_for_wavfile, 0, None, None)

    ###########################################################################
    # Import json data
    ###########################################################################
    CONFIG_DATAS = get_results_binary_relevance.import_predict_configuration_json(predictions_cfg_json)
    LABEL_NAMES = list(CONFIG_DATAS.keys())

    ###########################################################################
    # Implementing using the keras usual training techinque
    ###########################################################################

    CLF2_TRAIN_PREDICTION = []
    CLF2_TRAIN_PREDICTION_PROB = []

    for each_embedding in [EMBEDDINGS]:

        prediction_probs, prediction_rounded = predict_on_embedding(embedding = each_embedding,
                                                                    label_names = LABEL_NAMES,
                                                                    config_datas = CONFIG_DATAS)

        CLF2_TRAIN_PREDICTION_PROB.append(prediction_probs)
        CLF2_TRAIN_PREDICTION.append(prediction_rounded)

    ###########################################################################
        # Print results
    ###########################################################################
    results = pd.DataFrame(np.array(CLF2_TRAIN_PREDICTION_PROB), columns=LABEL_NAMES)
    print(results)


if __name__ == "__main__":

    ###########################################################################
    # Description and Help
    ###########################################################################
    DESCRIPTION = 'Gets the predictions of sounds. \n \
                   Input base dataframe file path \
                   with feature (Embeddings) and labels_name column in it.'

    ###########################################################################
    # Parsing the inputs given
    ###########################################################################
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()

    REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
    REQUIRED_NAMED.add_argument('-predictions_cfg_json',
                                '--predictions_cfg_json',
                                help='Input json configuration file for predictions output',
                                required=True)
    REQUIRED_NAMED.add_argument('-path_for_wavfile',
                                '--path_for_wavfile',
                                action='store',
                                help='Input the path for 10second wavfile',
                                required=True)

    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    PARSED_ARGS = ARGUMENT_PARSER.parse_args()

    main(predictions_cfg_json = PARSED_ARGS.predictions_cfg_json,
         path_for_wavfile = PARSED_ARGS.path_for_wavfile)
