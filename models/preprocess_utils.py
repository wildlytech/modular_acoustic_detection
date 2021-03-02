import json
import numpy as np
import pandas as pd
import pickle
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
import glob


def read_json_file(filepath):
    with open(filepath, "r") as f:
        file = json.load(f)
    return file


def subsample_dataframe(dataframe, subsample):
    """
    Subsample examples from the dataframe
    """
    if subsample is not None:
        if subsample > 0:
            # If subsample is less than size of dataframe, then
            # don't allow replacement when sampling
            # Otherwise, the intention is to oversample
            dataframe = dataframe.sample(subsample, replace=(subsample > dataframe.shape[0]))
        else:
            dataframe = pd.DataFrame([], columns=dataframe.columns)

    return dataframe


def split_and_subsample_dataframe(dataframe, validation_split, subsample):
    """
    Perform validation split and sub/over sampling of dataframe
    Returns train and test dataframe
    """

    # split before sub/oversampling to ensure there is no leakage between train and test sets
    test_size = int(validation_split * dataframe.shape[0])
    train_size = dataframe.shape[0] - test_size

    if test_size == 0:
        train_df = dataframe
        test_df = pd.DataFrame({}, columns=dataframe.columns)
    elif train_size == 0:
        train_df = pd.DataFrame({}, columns=dataframe.columns)
        test_df = dataframe
    else:
        train_df, test_df = train_test_split(dataframe,
                                             test_size=test_size,
                                             random_state=42)

    test_subsample = int(subsample * validation_split)
    train_subsample = subsample - test_subsample

    if (train_df.shape[0] == 0) and (train_subsample > 0):
        print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)
        assert (False)
    else:
        train_df = subsample_dataframe(train_df, train_subsample)

    if (test_df.shape[0] == 0) and (test_subsample > 0):
        print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)
        assert (False)
    else:
        test_df = subsample_dataframe(test_df, test_subsample)

    return train_df, test_df


def get_select_vector(dataframe, label_filter_arr):
    """
    Get the boolean select vector on the dataframe from the label filter
    """
    return dataframe['labels_name'].apply(lambda arr: np.any([x.lower() in label_filter_arr for x in arr]))


def import_dataframes(dataframe_file_list,
                      positive_label_filter_arr,
                      negative_label_filter_arr,
                      validation_split,
                      model="binary"):
    """
    Iterate through each pickle file and import a subset of the dataframe
    """

    # All entries that have a pattern path need special handling
    # Specifically, all the files that match the pattern path
    # need to be determined
    pattern_file_dicts = [x for x in dataframe_file_list if "patternPath" in list(x.keys())]

    # Keep just the entries that don't have a pattern path
    # We'll add entries for each pattern path separately
    dataframe_file_list = [x for x in dataframe_file_list if "patternPath" not in list(x.keys())]

    # Expand out each pattern path entry into individual ones with fixed paths
    for input_file_dict in pattern_file_dicts:
        pattern_path = input_file_dict["patternPath"]

        # Get list of any exclusions that should be made
        if "excludePaths" in list(input_file_dict.keys()):
            exclude_paths = input_file_dict["excludePaths"]

            if exclude_paths is None:
                exclude_paths = []
            elif type(exclude_paths) != list:
                exclude_paths = [exclude_paths]
        else:
            exclude_paths = []

        # Make excluded paths a set for fast lookup
        exclude_paths = set(exclude_paths)

        # Search for all paths that match this pattern
        search_results = glob(pattern_path)

        # If path is in the excluded paths, then ignore it
        search_results = [x for x in search_results if x not in exclude_paths]

        if len(search_results) == 0:
            print(Fore.RED, "No file matches pattern criteria:", pattern_path, "Excluded Paths:", exclude_paths,
                  Style.RESET_ALL)
            assert (False)

        for path in search_results:
            # Add an entry with fixed path
            fixed_path_dict = input_file_dict.copy()

            # Remove keys that are used for pattern path entry
            # Most other keys should be the same as the pattern path entry
            del fixed_path_dict["patternPath"]
            if "excludePaths" in list(fixed_path_dict.keys()):
                del fixed_path_dict["excludePaths"]

            # Make it look like a fixed path entry
            fixed_path_dict["path"] = path
            dataframe_file_list += [fixed_path_dict]

    # Proceed with importing all dataframe files
    list_of_train_dataframes = []
    list_of_test_dataframes = []
    if model == "binary":
        for input_file_dict in dataframe_file_list:

            assert ("patternPath" not in list(input_file_dict.keys()))
            assert ("path" in list(input_file_dict.keys()))

            print("Importing", input_file_dict["path"], "...")

            with open(input_file_dict["path"], 'rb') as file_obj:

                # Load the file
                df = pickle.load(file_obj)

                # Filtering the sounds that are exactly 10 seconds
                # Examples should be exactly 10 seconds. Anything else
                # is not a valid input to the model
                df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]

                # Only use examples that have a label in label filter array
                positive_example_select_vector = get_select_vector(df, positive_label_filter_arr)
                positive_examples_df = df.loc[positive_example_select_vector]

                # This ensures there no overlap between positive and negative examples
                negative_example_select_vector = ~positive_example_select_vector
                if negative_label_filter_arr is not None:
                    # Exclude even further examples that don't fall into the negative label filter
                    negative_example_select_vector &= get_select_vector(df, negative_label_filter_arr)
                negative_examples_df = df.loc[negative_example_select_vector]

                # No longer need df after this point
                del df

                train_positive_examples_df, test_positive_examples_df = \
                    split_and_subsample_dataframe(dataframe=positive_examples_df,
                                                  validation_split=validation_split,
                                                  subsample=input_file_dict["positiveSubsample"])
                del positive_examples_df

                train_negative_examples_df, test_negative_examples_df = \
                    split_and_subsample_dataframe(dataframe=negative_examples_df,
                                                  validation_split=validation_split,
                                                  subsample=input_file_dict["negativeSubsample"])
                del negative_examples_df

                # append to overall list of examples
                list_of_train_dataframes += [train_positive_examples_df, train_negative_examples_df]
                list_of_test_dataframes += [test_positive_examples_df, test_negative_examples_df]

        train_df = pd.concat(list_of_train_dataframes, ignore_index=True)
        test_df = pd.concat(list_of_test_dataframes, ignore_index=True)

        print("Import done.")

        return train_df, test_df
    else:

        for input_file_dict in dataframe_file_list:
            assert ("patternPath" not in list(input_file_dict.keys()))
            assert ("path" in list(input_file_dict.keys()))

            print("Importing", input_file_dict["path"], "...")

            with open(input_file_dict["path"], 'rb') as file_obj:
                # Load the file

                df = pickle.load(file_obj)

                # Filtering the sounds that are exactly 10 seconds
                # Examples should be exactly 10 seconds. Anything else
                # is not a valid input to the model
                df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]

                train_file_examples_df, test_file_examples_df = \
                    split_and_subsample_dataframe(dataframe=df,
                                                  validation_split=validation_split,
                                                  subsample=input_file_dict["subsample"])

                # append to overall list of examples
                list_of_train_dataframes.append(train_file_examples_df)
                list_of_test_dataframes.append(test_file_examples_df)

        train_df = pd.concat(list_of_train_dataframes, ignore_index=True)
        test_df = pd.concat(list_of_test_dataframes, ignore_index=True)

        print("Import done.")

        return train_df, test_df
