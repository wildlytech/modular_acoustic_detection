"""
Testing a BR - Model
"""
import argparse
import json
import pandas as pd
import numpy as np
from tensorflow.compat.v1.keras import backend as K
from predictions.binary_relevance_model import generate_before_predict_BR
from . import multilabel_pred


def predict_on_embedding(embedding, config_datas):
    '''
    Predict on single embedding for audio clip
    '''

    config_data = config_datas

    # Clear keras session before all predictions with model
    K.clear_session()

    model = multilabel_pred.load_model(config_data["networkCfgJson"],
                                       config_data["train"]["outputWeightFile"])

    ###########################################################################
    # Predict on test data
    ###########################################################################

    if embedding.shape[0] < 10:
        # Pad by repeating the last second in embedding till we get to 10 seconds
        padding = np.ones((10 - embedding.shape[0], embedding.shape[1])) * embedding[-1, :]
        embedding = np.vstack([embedding, padding])

    # If the clip is longer then 10 seconds, then predict on multiple 10-second
    # windows within the clip. Take the max probability across all windows.
    prediction_probs = np.zeros(len(config_data["labels"]))
    for index in np.arange(10, embedding.shape[0] + 1):
        windowed_embedding = embedding[(index - 10):index, :].reshape((1, 10 * 128, 1))
        window_pred_prob = model.predict(windowed_embedding)
        prediction_probs = np.array([prediction_probs, window_pred_prob.ravel()]).max(axis=0)

    # Clear keras session after all predictions with model
    K.clear_session()

    prediction_rounded = np.round(prediction_probs)
    prediction_probs = prediction_probs * 100

    return prediction_probs, prediction_rounded


def read_config(filepath):
    with open(filepath, "r") as f:
        config = json.load(f)
    return config


def main(predictions_cfg_json, path_for_wavfile):

    ###########################################################################
    # read the dataframe with feature and labels_name column
    ###########################################################################
    EMBEDDINGS = generate_before_predict_BR.main(path_for_wavfile, 0, None, None)

    ###########################################################################
    # Import json data
    ###########################################################################
    CONFIG_DATAS = [read_config(file) for file in read_config(predictions_cfg_json)]

    ###########################################################################
    # Implementing using the keras usual training technique
    ###########################################################################
    colnames = []

    CLF2_TRAIN_PREDICTION = []
    CLF2_TRAIN_PREDICTION_PROB = []
    for each_embedding in [EMBEDDINGS]:
        for data in CONFIG_DATAS:

            for label in data["labels"]:
                colnames.append(label["aggregatePositiveLabelName"])

            prediction_probs, prediction_rounded = predict_on_embedding(embedding=each_embedding,
                                                                        config_datas=data)

            CLF2_TRAIN_PREDICTION_PROB.append(prediction_probs)
            CLF2_TRAIN_PREDICTION.append(prediction_rounded)

    ###########################################################################
    # Print results
    ###########################################################################
    results = pd.DataFrame(np.array(CLF2_TRAIN_PREDICTION_PROB), columns=colnames)

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
                                help='Input the path for 10 second wavfile',
                                required=True)

    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    PARSED_ARGS = ARGUMENT_PARSER.parse_args()

    main(predictions_cfg_json=PARSED_ARGS.predictions_cfg_json,
         path_for_wavfile=PARSED_ARGS.path_for_wavfile)
