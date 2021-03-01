
import argparse
import pickle


def pickle_to_csv(path_to_dataframe,
                  path_to_output_csv_file,
                  drop_features=True):
    """
    Convert a pickle file to csv format
    """

    with open(path_to_dataframe, "rb") as f:
        df = pickle.load(f)

        if drop_features:
            df.drop('features', axis=1, inplace=True)

        df.to_csv(path_to_output_csv_file)


########################################################################
# Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'Convert pickle dataframe file to csv'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_ARGUMENTS = PARSER._action_groups.pop()
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-d', '--dataframe',
                                    action='store',
                                    help='Path to feature dataframe '
                                         'pickle file',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-c', '--csv',
                                    action='store',
                                    help='Path to csv file to write dataframe',
                                    required=True)
    OPTIONAL_ARGUMENTS.add_argument('-ddf', '--dont_drop_features',
                                    action='store_true',
                                    help='Default drop features column '
                                         'since it can be very large. '
                                         'Set this to not drop features '
                                         'column')

    PARSER._action_groups.append(OPTIONAL_ARGUMENTS)
    RESULT = PARSER.parse_args()

    pickle_to_csv(path_to_dataframe=RESULT.dataframe,
                  path_to_output_csv_file=RESULT.csv,
                  drop_features=(not RESULT.dont_drop_features))
