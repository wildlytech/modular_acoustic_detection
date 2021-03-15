
import argparse
import pickle


def remove_rows_from_dataframe(path_to_dataframe,
                               path_to_blacklist_file,
                               blacklist_item_suffix,
                               id_column,
                               output_file_name):
    """
    Given a path to a pickle frame with features for each 10 second chunk AND
    a path to a csv file with all the labels for the original clips, add labels to
    the feature dataframe and write out the pickle file.
    """
    dataframe = pickle.load(open(path_to_dataframe, "rb"))

    with open(path_to_blacklist_file, 'r') as f:
        blacklist = f.readlines()
        blacklist = [x + blacklist_item_suffix for x in blacklist]

    in_blacklist = dataframe[id_column].apply(lambda x: x in blacklist)
    print("Removing rows with IDs:", dataframe.loc[in_blacklist, id_column].tolist())

    dataframe = dataframe.loc[in_blacklist == 0]
    dataframe.reset_index(drop=True, inplace=True)

    with open(output_file_name, 'wb') as file_obj:
        pickle.dump(dataframe, file_obj)

    print("Wrote", output_file_name)

    return dataframe


########################################################################
# Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'Remove rows from dataframe pickle file using entries in blacklist file'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_ARGUMENTS = PARSER._action_groups.pop()
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-d', '--dataframe',
                                    action='store',
                                    help='Path to dataframe pickle file',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-b', '--blacklist_file',
                                    action='store',
                                    help='Path to file with blacklist ids (e.g. file ids)',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-c', '--id_column',
                                    action='store',
                                    help='ID column in dataframe pickle file',
                                    required=True)
    OPTIONAL_ARGUMENTS.add_argument('-s', '--blacklist_item_suffix',
                                    action='store',
                                    help='Suffix to add to entry in blacklist file',
                                    default='')
    OPTIONAL_ARGUMENTS.add_argument('-o', '--output_file_name',
                                    action='store',
                                    help='Path to output dataframe pickle file with rows removed. '
                                         'Otherwise output will be <dataframe> + \'_blacklist_removed.pkl\'')
    PARSER._action_groups.append(OPTIONAL_ARGUMENTS)
    RESULT = PARSER.parse_args()

    output_file_name = RESULT.output_file_name
    if output_file_name is None:
        assert(RESULT.dataframe.endswith(".pkl"))
        output_file_name = RESULT.dataframe[:-4] + "_blacklist_removed.pkl"

    remove_rows_from_dataframe(path_to_dataframe=RESULT.dataframe,
                               path_to_blacklist_file=RESULT.blacklist_file,
                               blacklist_item_suffix=RESULT.blacklist_item_suffix,
                               id_column=RESULT.id_column,
                               output_file_name=output_file_name)
