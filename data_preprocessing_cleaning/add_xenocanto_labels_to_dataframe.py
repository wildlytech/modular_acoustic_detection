
import argparse
from colorama import Fore, Style
import pandas as pd
import pickle
import re


def add_labels_to_dataframe(path_to_feature_dataframe, path_to_label_csv_file):
    """
    Given a path to a pickle frame with features for each 10 second chunk AND
    a path to a csv file with all the labels for the original clips, add labels to
    the feature dataframe and write out the pickle file.
    """
    feature_dataframe = pickle.load(open(path_to_feature_dataframe, "rb"))
    label_dataframe = pd.read_csv(path_to_label_csv_file, index_col='XenoCanto_ID')

    labels_name_column = []
    for index, split_wav_filename in enumerate(feature_dataframe.wav_file):
        # Remove the chunk ID from the name to get the XenoCanto ID
        orig_wav_filename = "_".join(split_wav_filename.split('_')[:-1])
        row = label_dataframe.loc[orig_wav_filename]

        labels_name = []

        if "gen" in label_dataframe.columns and \
           "sp" in label_dataframe.columns and \
           "ssp" in label_dataframe.columns:

            gen = row["gen"]
            if not pd.isnull(gen):
                gen = gen.lower()
                labels_name.append(gen)

                sp = row["sp"]
                if not pd.isnull(sp):
                    sp = sp.lower()
                    labels_name.append("{0} {1}".format(gen, sp))

                    ssp = row["ssp"]
                    if not pd.isnull(ssp):
                        ssp = ssp.lower()
                        labels_name.append("{0} {1} {2}".format(gen, sp, ssp))

        else:

            # Use the old column format from xenocanto_scrape
            name = label_dataframe['Common name/Scientific'].loc[orig_wav_filename].lower()

            match = re.match(r"(.*) \((.+)\)", name)
            if match:
                # Each entry in labels name array is an array of labels
                sci_name = match.group(2)
            else:
                # If it doesn't fit the format, it must not have a common english name.
                # The entire name must be the scientific name
                sci_name = name

            sci_name_split = sci_name.split(' ')

            # Scientific name has format: [GENUS] or [GENUS SPECIES] or [GENUS SPECIES SUBSPECIES]
            # Examples with longer names can also carry shorter less specific names (e.g. subspecies
            # can also be labeled as species or genus). Make sure the examples is labeled with all
            # possible names. This will ensure that at leasrt one of the labels exist in the ontology even
            # when the more descriptive name does not exist
            for length in range(1, min(3, len(sci_name_split)) + 1):
                labels_name.append(' '.join(sci_name_split[:length]))

            # Check if name format is not what we expect
            print_warning = False
            if len(sci_name_split) > 3:
                # Go ahead and add full scientific name as a label anyway
                labels_name.append(sci_name)
                print_warning = True

            # There should only be lower case alphabetic letters in scientific name
            for word in sci_name_split:
                if not re.match("[a-z]+", word):
                    print_warning = True

            if print_warning:
                print(Fore.RED,
                      "Warning: Example {0} has scientific name in unexpected format: \"{1}\"".format(split_wav_filename, sci_name),
                      Style.RESET_ALL)

        # Add array of labels as entry for this example in labels name column
        labels_name_column.append(labels_name)

    feature_dataframe['labels_name'] = labels_name_column

    assert(path_to_feature_dataframe.endswith(".pkl"))
    path_to_full_dataframe = path_to_feature_dataframe[:-4] + "_with_labels.pkl"

    with open(path_to_full_dataframe, 'wb') as file_obj:
        pickle.dump(feature_dataframe, file_obj)

    return feature_dataframe


########################################################################
# Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'Add a labels_name column to xenocanto dataframe from csv file'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_ARGUMENTS = PARSER._action_groups.pop()
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-c', '--csv',
                                    action='store',
                                    help='Path to csv file with xenocanto information',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-d', '--dataframe',
                                    action='store',
                                    help='Path to dataframe pickle file',
                                    required=True)
    PARSER._action_groups.append(OPTIONAL_ARGUMENTS)
    RESULT = PARSER.parse_args()

    add_labels_to_dataframe(path_to_feature_dataframe=RESULT.dataframe,
                            path_to_label_csv_file=RESULT.csv)
