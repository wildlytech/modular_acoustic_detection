
import argparse
import numpy as np
import pandas as pd
import pickle


def add_csv_labels_to_dataframe(path_to_feature_dataframe,
                                path_to_label_csv_file,
                                dataframe_join_column,
                                csv_join_column,
                                column_names,
                                append,
                                output_file_name):
    """
    Given a path to a pickle frame with features for each 10 second chunk AND
    a path to a csv file with all the labels for the original clips, add labels to
    the feature dataframe and write out the pickle file.
    """
    feature_dataframe = pickle.load(open(path_to_feature_dataframe, "rb"))
    feature_dataframe.index = feature_dataframe[dataframe_join_column]

    csv_dataframe = pd.read_csv(path_to_label_csv_file)
    csv_dataframe.index = csv_dataframe[csv_join_column]

    # Assumption is that all the same rows are represented in both dataframes
    assert(feature_dataframe.shape[0] == csv_dataframe.shape[0])

    # If labels_name column does not already exist or we need to write over it,
    # then create the column
    if ('labels_name' not in feature_dataframe.columns) or (not append):
        feature_dataframe['labels_name'] = [[] for i in range(feature_dataframe.shape[0])]

    for column_name in column_names:
        assert(csv_dataframe[column_name].apply(lambda x: type(x) == str).all())
        feature_dataframe['labels_name'] = feature_dataframe['labels_name'] + \
            csv_dataframe[column_name].apply(lambda x: [x])

    feature_dataframe.index = np.arange(feature_dataframe.shape[0])
    with open(output_file_name, 'wb') as file_obj:
        pickle.dump(feature_dataframe, file_obj)

    print("Wrote", output_file_name)

    return feature_dataframe


########################################################################
# Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'Add a labels_name column to dataframe pickle file from specified columns in csv file'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_ARGUMENTS = PARSER._action_groups.pop()
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-d', '--dataframe',
                                    action='store',
                                    help='Path to feature dataframe pickle file',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-csv', '--csv',
                                    action='store',
                                    help='Path to csv file with label information',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-cols', '--column_names',
                                    nargs='+',
                                    help='Column(s) in csv file to add to labels_name column in feature dataframe',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-dj', '--dataframe_join_column',
                                    action='store',
                                    help='Column in dataframe pickle file to join on',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-cj', '--csv_join_column',
                                    action='store',
                                    help='Column in csv file to join on',
                                    required=True)
    OPTIONAL_ARGUMENTS.add_argument('-a', '--append',
                                    action='store_true',
                                    help='Append to existing labels in labels_name column. Otherwise replace.')
    OPTIONAL_ARGUMENTS.add_argument('-o', '--output_file_name',
                                    action='store',
                                    help='Path to output labeled dataframe pickle file. '
                                         'Otherwise output will be <dataframe> + \'_with_labels.pkl\'')
    PARSER._action_groups.append(OPTIONAL_ARGUMENTS)
    RESULT = PARSER.parse_args()

    output_file_name = RESULT.output_file_name
    if output_file_name is None:
        assert(RESULT.dataframe.endswith(".pkl"))
        output_file_name = RESULT.dataframe[:-4] + "_with_labels.pkl"

    add_csv_labels_to_dataframe(path_to_feature_dataframe=RESULT.dataframe,
                                path_to_label_csv_file=RESULT.csv,
                                dataframe_join_column=RESULT.dataframe_join_column,
                                csv_join_column=RESULT.csv_join_column,
                                column_names=RESULT.column_names,
                                append=RESULT.append,
                                output_file_name=output_file_name)
