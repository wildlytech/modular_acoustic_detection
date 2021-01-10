"""
Testing a Multi label Model
"""
# Import the necessary functions and libraries
import argparse
import json
from tensorflow.compat.v1.keras.models import model_from_json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss

from youtube_audioset import get_recursive_sound_names


def import_predict_configuration_json(predictions_cfg_json):
    """
    Import and process nested json data from predictions configuration file.

    Returns a dictionary with all configuration json data for all labels.
    """
    config_data_dict = {}

    with open(predictions_cfg_json) as predictions_json_file_obj:

        list_of_config_files = json.load(predictions_json_file_obj)

        # Each entry in json file is a path to model cfg file for one label
        for filepath in list_of_config_files:

            with open(filepath) as json_data_obj:

                config_data = json.load(json_data_obj)

                # Model only supports using audio set as main ontology
                assert(config_data["ontology"]["useYoutubeAudioSet"])

                # List of paths to json files that will be used to extend
                # existing youtube ontology
                ontologyExtFiles = config_data["ontology"]["extension"]

                # If single file or null, then convert to list
                if ontologyExtFiles is None:
                    ontologyExtFiles = []
                elif type(ontologyExtFiles) != list:
                    ontologyExtFiles = [ontologyExtFiles]

                # Update extension paths in dictionary
                config_data["ontology"]["extension"] = ontologyExtFiles

                model_name = config_data["name"]
                config_data_dict[model_name] = config_data

    return config_data_dict


def load_model(networkCfgJson, weightFile):
    """
    Returns the keras model using the network json configuration file and
    weight file.
    """

    # load json and create model
    json_file = open(networkCfgJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(weightFile)

    return model


def main(predictions_cfg_json,
         path_for_dataframe_with_features,
         save_misclassified_examples=None,
         path_to_save_prediction_csv=None):

    ###########################################################################
    # Import json data
    ###########################################################################

    CONFIG_DATAS = import_predict_configuration_json(predictions_cfg_json)

    ###########################################################################
    # read the dataframe with feature and labels_name column
    ###########################################################################

    print("Importing Data...")
    with open(path_for_dataframe_with_features, "rb") as file_obj:
        DATA_FRAME = pickle.load(file_obj)
        DATA_FRAME.index = list(range(0, DATA_FRAME.shape[0]))

    IS_DATAFRAME_LABELED = 'labels_name' in DATA_FRAME.columns

    if IS_DATAFRAME_LABELED:
        print("Categorizing labels in dataframe...")
        #######################################################################
        # Check if labels fall into positive label designation
        #######################################################################
        LABELS_BINARIZED = pd.DataFrame()

        for model_name in list(CONFIG_DATAS.keys()):
            positiveLabels = {}
            config_data = CONFIG_DATAS[model_name]
            for label in config_data["labels"]:
                positiveLabels[label["aggregatePositiveLabelName"]] = \
                    get_recursive_sound_names(designated_sound_names=label["positiveLabels"],
                                              path_to_ontology="./",
                                              ontology_extension_paths=config_data["ontology"]["extension"])

            for key in positiveLabels.keys():
                pos_lab = positiveLabels[key]
                binarized_op_column = 1.0 * DATA_FRAME['labels_name'].apply( \
                    lambda arr: np.any([x.lower() in pos_lab for x in arr]))

                LABELS_BINARIZED[key] = binarized_op_column
        target_cols = LABELS_BINARIZED.columns

    ###########################################################################
    # Filtering the sounds that are exactly 10 seconds
    ###########################################################################
    DF_TEST = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]

    if IS_DATAFRAME_LABELED:

        LABELS_FILTERED = LABELS_BINARIZED.loc[DF_TEST.index, :]

    ###########################################################################
    # preprocess the data into required structure
    ###########################################################################

    X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())

    ###########################################################################
    # reshaping the test data so as to align with input for model
    ###########################################################################
    CLF2_TEST = X_TEST.reshape((-1, 1280, 1))

    ###########################################################################
    # Implementing using the keras usual prediction technique
    ###########################################################################

    for model_name in list(CONFIG_DATAS.keys()):

        config_data = CONFIG_DATAS[model_name]

        MODEL = load_model(config_data["networkCfgJson"], config_data["train"]["outputWeightFile"])

        print(("\nLoaded " + model_name + " model from disk"))

        #######################################################################
        # Predict on test data
        #######################################################################

        CLF2_TEST_PREDICTION_PROB = MODEL.predict(CLF2_TEST)

        CLF2_TEST_PREDICTION = CLF2_TEST_PREDICTION_PROB.round()

        pred_args = CLF2_TEST_PREDICTION_PROB.argmax(axis=1)

        prob_colnames = [label_name + "_Probability" for label_name in target_cols]
        pred_colnames = [label_name + "_Prediction" for label_name in target_cols]

        DF_TEST_PRED = pd.concat([pd.DataFrame(CLF2_TEST_PREDICTION_PROB, columns=prob_colnames),
                                  pd.DataFrame(CLF2_TEST_PREDICTION, columns=pred_colnames)],
                                 axis=1)
        DF_TEST = pd.concat([DF_TEST.reset_index(drop=True), DF_TEST_PRED], axis=1)

        if IS_DATAFRAME_LABELED:
            ###################################################################
            # Target for the test labels
            ###################################################################
            CLF2_TEST_TARGET = LABELS_FILTERED.values

            gt_args = CLF2_TEST_TARGET.argmax(axis=1)
            ###################################################################
            # To get the Misclassified examples
            ###################################################################
            actual_colnames = [label_name + "_Actual" for label_name in target_cols]

            CLF2_TEST_TARGET = pd.DataFrame(CLF2_TEST_TARGET,
                                            columns=actual_colnames).reset_index(drop=True)
            DF_TEST = pd.concat([DF_TEST, CLF2_TEST_TARGET], axis=1)

            MISCLASSIFED_ARRAY = (CLF2_TEST_PREDICTION != CLF2_TEST_TARGET).any(axis=1)
            print('\nMisclassified number of examples for ' + model_name + " :", \
                  MISCLASSIFED_ARRAY.sum())

            ###################################################################
            # misclassified examples are to be saved
            ###################################################################
            if save_misclassified_examples:
                misclassified_pickle_file = save_misclassified_examples + \
                    "_misclassified_examples_multilabel_model_" + \
                    model_name.replace(' ', '_') + ".pkl"
                with open(misclassified_pickle_file, "wb") as f:
                    pickle.dump(DF_TEST.loc[MISCLASSIFED_ARRAY].drop(["features"], axis=1), f)

            ###################################################################
            # Print confusion matrix and classification_report
            ###################################################################
            print('Confusion Matrix for ' + model_name)
            print('============================================')
            for i in range(CLF2_TEST_TARGET.shape[1]):
                print("Confusion matrix for", target_cols[i])
                a = CLF2_TEST_TARGET.iloc[:, i].values
                b = CLF2_TEST_PREDICTION[:, i]
                RESULT_ = confusion_matrix(a, b)
                print(RESULT_)

            ###################################################################
            # print classification report
            ###################################################################
            print('Classification Report for ' + model_name)
            print('============================================')
            CL_REPORT = classification_report(gt_args,
                                              pred_args)
            print(CL_REPORT)

            ###################################################################
            # calculate accuracy and hamming loss
            ###################################################################
            ACCURACY = accuracy_score(gt_args,
                                      pred_args)
            HL = hamming_loss(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION)
            print('Hamming Loss :', HL)
            print('Accuracy :', ACCURACY)

    ###########################################################################
    # save the prediction in pickle format
    ###########################################################################

    if path_to_save_prediction_csv:
        DF_TEST.drop(["features"], axis=1).to_csv(path_to_save_prediction_csv)


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
    REQUIRED_NAMED.add_argument('-path_for_dataframe_with_features',
                                '--path_for_dataframe_with_features',
                                action='store',
                                help='Input the path for dataframe(.pkl format) with features',
                                required=True)

    OPTIONAL_NAMED.add_argument('-save_misclassified_examples',
                                '--save_misclassified_examples',
                                action='store',
                                help='Input the directory to save the misclassified examples.\
                                      Not applicable when data is unlabeled.')
    OPTIONAL_NAMED.add_argument('-path_to_save_prediction_csv',
                                '--path_to_save_prediction_csv',
                                action='store',
                                help='Input the path to save predictions (.csv)')

    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    PARSED_ARGS = ARGUMENT_PARSER.parse_args()

    main(predictions_cfg_json=PARSED_ARGS.predictions_cfg_json,
         path_for_dataframe_with_features=PARSED_ARGS.path_for_dataframe_with_features,
         save_misclassified_examples=PARSED_ARGS.save_misclassified_examples,
         path_to_save_prediction_csv=PARSED_ARGS.path_to_save_prediction_csv)
