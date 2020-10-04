"""
Testing a Binary Relevance Model
"""
#Import the necessary functions and libraries
import argparse
import json
from keras.models import model_from_json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
import sys
sys.path.insert(0, "../../")
from youtube_audioset import get_recursive_sound_names

def import_predict_configuration_json(predictions_cfg_json):
    """
    Import and process nested json data from predictions configuration file.

    Returns a dictionary with all configuration json data for all labels.
    """
    config_data_dict  = {}

    with open(predictions_cfg_json) as predictions_json_file_obj:

        list_of_config_files = json.load(predictions_json_file_obj)

        # Each entry in json file is a path to model cfg file for one label
        for filepath in list_of_config_files:

            directory_of_filepath = '/'.join(filepath.split('/')[:-1]) + '/'

            with open(filepath) as json_data_obj:

                config_data = json.load(json_data_obj)

                config_data["networkCfgJson"] = directory_of_filepath + \
                                                config_data["networkCfgJson"]
                config_data["train"]["outputWeightFile"] = directory_of_filepath + \
                                                           config_data["train"]["outputWeightFile"]

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

                # All paths to ontology extension files are relative to the location of the
                # model configuration file.
                ontologyExtFiles = [directory_of_filepath + x for x in ontologyExtFiles]

                # Update extension paths in dictionary
                config_data["ontology"]["extension"] = ontologyExtFiles

                label_name = '['+config_data["aggregatePositiveLabelName"] + ']Vs[' + \
                             config_data["aggregateNegativeLabelName"] + ']'

                config_data_dict[label_name] = config_data

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
         save_misclassified_examples = None,
         path_to_save_prediction_csv = None):

    ##############################################################################
              # Import json data
    ##############################################################################

    CONFIG_DATAS = import_predict_configuration_json(predictions_cfg_json)

    ##############################################################################
          # read the dataframe with feature and labels_name column
    ##############################################################################

    print("Importing Data...")
    with open(path_for_dataframe_with_features, "rb") as file_obj:
        DATA_FRAME = pickle.load(file_obj)
        DATA_FRAME.index = list(range(0, DATA_FRAME.shape[0]))

    IS_DATAFRAME_LABELED = 'labels_name' in DATA_FRAME.columns

    if IS_DATAFRAME_LABELED:
        print("Categorizing labels in dataframe...")
        ##############################################################################
                # Check if labels fall into positive label designation
        ##############################################################################
        LABELS_BINARIZED = pd.DataFrame()

        for label_name in list(CONFIG_DATAS.keys()):

            config_data = CONFIG_DATAS[label_name]

            positiveLabels = get_recursive_sound_names(designated_sound_names = config_data["positiveLabels"],
                                                       path_to_ontology = "../../",
                                                       ontology_extension_paths = config_data["ontology"]["extension"])

            LABELS_BINARIZED[label_name] = 1.0 * DATA_FRAME['labels_name'].apply( \
                                           lambda arr: np.any([x.lower() in positiveLabels for x in arr]))

    ##############################################################################
          # Filtering the sounds that are exactly 10 seconds
    ##############################################################################
    DF_TEST = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]

    if IS_DATAFRAME_LABELED:
        LABELS_FILTERED = LABELS_BINARIZED.loc[DF_TEST.index, :]


    ##############################################################################
          # preprecess the data into required structure
    ##############################################################################
    X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
    X_TEST_STANDARDIZED = X_TEST / 255


    ##############################################################################
      # reshaping the test data so as to align with input for model
    ##############################################################################
    CLF2_TEST = X_TEST.reshape((-1, 1280, 1))


    ##############################################################################
        # Implementing using the keras usual prediction technique
    ##############################################################################

    for label_name in list(CONFIG_DATAS.keys()):

        config_data = CONFIG_DATAS[label_name]

        MODEL = load_model(config_data["networkCfgJson"], config_data["train"]["outputWeightFile"])

        print(("\nLoaded " + label_name + " model from disk"))

        ##############################################################################
              # Predict on test data
        ##############################################################################
        CLF2_TEST_PREDICTION_PROB = MODEL.predict(CLF2_TEST).ravel()
        CLF2_TEST_PREDICTION = CLF2_TEST_PREDICTION_PROB.round()

        # Add results to data frame
        DF_TEST.insert(len(DF_TEST.columns), label_name+"_Probability", CLF2_TEST_PREDICTION_PROB)
        DF_TEST.insert(len(DF_TEST.columns), label_name+"_Prediction", CLF2_TEST_PREDICTION)


        if IS_DATAFRAME_LABELED:
            ##############################################################################
                    # Target for the test labels
            ##############################################################################
            CLF2_TEST_TARGET = LABELS_FILTERED[label_name].values
            print('Target shape:', CLF2_TEST_TARGET.shape)

            ##############################################################################
                    # To get the Misclassified examples
            ##############################################################################
            DF_TEST.insert(len(DF_TEST.columns), label_name+'_Actual', CLF2_TEST_TARGET)
            MISCLASSIFED_ARRAY = CLF2_TEST_PREDICTION != CLF2_TEST_TARGET
            print('\nMisclassified number of examples for '+ label_name + " :", \
                  DF_TEST.loc[MISCLASSIFED_ARRAY].shape[0])


            ##############################################################################
                    #  misclassified examples are to be saved
            ##############################################################################
            if save_misclassified_examples:
                misclassified_pickle_file = save_misclassified_examples + \
                              "misclassified_examples_br_model_"+label_name+".pkl"
                with open(misclassified_pickle_file, "w") as f:
                    pickle.dump(DF_TEST[MISCLASSIFED_ARRAY].drop(["features"], axis=1), f)


            ##############################################################################
                    # Print confusion matrix and classification_report
            ##############################################################################
            print('Confusion Matrix for '+ label_name)
            print('============================================')
            RESULT_ = confusion_matrix(CLF2_TEST_TARGET,
                                       CLF2_TEST_PREDICTION)
            print(RESULT_)


            ##############################################################################
                  # print classification report
            ##############################################################################
            print('Classification Report for '+ label_name)
            print('============================================')
            CL_REPORT = classification_report(CLF2_TEST_TARGET,
                                              CLF2_TEST_PREDICTION)
            print(CL_REPORT)


            ##############################################################################
                  # calculate accuracy and hamming loss
            ##############################################################################
            ACCURACY = accuracy_score(CLF2_TEST_TARGET,
                                      CLF2_TEST_PREDICTION)
            HL = hamming_loss(CLF2_TEST_TARGET, CLF2_TEST_PREDICTION)
            print('Hamming Loss :', HL)
            print('Accuracy :', ACCURACY)

    ##############################################################################
          # save the prediction in pickle format
    ##############################################################################

    if path_to_save_prediction_csv:
        DF_TEST.drop(["features"], axis=1).to_csv(path_to_save_prediction_csv)

if __name__ == "__main__":

    ##############################################################################
            # Description and Help
    ##############################################################################
    DESCRIPTION = 'Gets the predictions of sounds. \n \
                   Input base dataframe file path \
                   with feature (Embeddings) and labels_name column in it.'

    ##############################################################################
            # Parsing the inputs given
    ##############################################################################

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

    main(predictions_cfg_json = PARSED_ARGS.predictions_cfg_json,
         path_for_dataframe_with_features = PARSED_ARGS.path_for_dataframe_with_features,
         save_misclassified_examples = PARSED_ARGS.save_misclassified_examples,
         path_to_save_prediction_csv = PARSED_ARGS.path_to_save_prediction_csv)
