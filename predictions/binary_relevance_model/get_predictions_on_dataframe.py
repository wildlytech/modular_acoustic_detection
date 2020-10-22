"""
Predictions with a Binary Relevance Model
"""
import argparse
from . import get_results_binary_relevance

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
    REQUIRED_NAMED.add_argument('-results_in_csv',
                                '--results_in_csv',
                                action='store',
                                help='Input the path to save predictions (.csv)',
                                required=True)

    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    PARSED_ARGS = ARGUMENT_PARSER.parse_args()

    get_results_binary_relevance.main(predictions_cfg_json = PARSED_ARGS.predictions_cfg_json,
                                      path_for_dataframe_with_features = PARSED_ARGS.path_for_dataframe_with_features,
                                      save_misclassified_examples = None,
                                      path_to_save_prediction_csv = PARSED_ARGS.results_in_csv)
