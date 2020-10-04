
import argparse
from colorama import Fore, Style
import create_base_dataframe
from data_preprocessing_cleaning import mp3_stereo_to_wav_mono
from data_preprocessing_cleaning import split_wav_file
import generating_embeddings
from get_data import xenocanto_scrape
import glob
import os
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

        name = label_dataframe['Common name/Scientific'].loc[orig_wav_filename].lower()

        labels_name = []

        match = re.match("(.*) \((.+)\)", name)
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
        for length in range(1,min(3,len(sci_name_split))+1):
            labels_name.append(' '.join(sci_name_split[:length]))

        # Check if name format is not what we expect
        print_warning = False;
        if len(sci_name_split) > 3:
            # Go ahead and add full scientific name as a label anyway
            labels_name.append(sci_name)
            print_warning = True

        # There should only be lower case alphabetic letters in scientific name
        for word in sci_name_split:
            if not re.match("[a-z]+", word):
                print_warning = True

        if print_warning:
            print(Fore.RED, \
                  "Warning: Example {0} has scientific name in unexpected format: \"{1}\"".format(split_wav_filename, sci_name), \
                  Style.RESET_ALL)

        # Add array of labels as entry for this example in labels name column
        labels_name_column.append(labels_name)

    feature_dataframe['labels_name'] = labels_name_column

    assert(path_to_feature_dataframe.endswith(".pkl"))
    path_to_full_dataframe = path_to_feature_dataframe[:-4] + "_with_labels.pkl"

    with open(path_to_full_dataframe, 'w') as file_obj:
        pickle.dump(feature_dataframe, file_obj)

    return feature_dataframe


def xenocanto_to_dataframe(bird_species,
                           output_path,
                           delete_mp3_files = False,
                           delete_wav_files = True,
                           delete_split_wav_files = True,
                           delete_embeddings = False):
    """
    Download xeno-canto sound files for particular bird species
    and perform entire data preprocessing pipeline to generate dataframe
    for training
    """

    if not output_path.endswith('/'):
        output_path += '/'

    # replace whitespace with underscore for bird_species name
    bird_species_name_ws = bird_species.replace(' ', '_')

    path_to_split_files = output_path + bird_species_name_ws + "/split_wav_files/"
    path_to_embeddings = output_path + bird_species_name_ws + "/embeddings/"
    path_to_write_dataframe = output_path + bird_species_name_ws + "/dataframe.pkl"

    print("Downloading audio files...")
    path_to_src_files, csv_filename = xenocanto_scrape.scrape(audio_files_path=output_path,
                                                              bird_species=bird_species)

    print("Converting mp3 to wav...")
    mp3_stereo_to_wav_mono.convert_files_directory(path_for_mp3_files=path_to_src_files,
                                                   path_to_save_wavfiles=path_to_src_files)

    print("Splitting wav files into 10 second clips...")
    split_wav_file.audio_split_directory(path_for_wavfiles = path_to_src_files,
                                         path_to_write_chunks = path_to_split_files,
                                         chunk_length_ms = 10000)

    print("Generating embeddings for each 10 second clip...")
    generating_embeddings.generate(path_to_write_embeddings = path_to_embeddings,
                                   path_to_wav_files = path_to_split_files)

    print("Building dataframe with features...")
    create_base_dataframe.create_new_dataframe(path_for_saved_embeddings=path_to_embeddings,
                                               path_to_write_dataframe=path_to_write_dataframe)

    print("Adding labels to dataframe...")
    add_labels_to_dataframe(path_to_feature_dataframe=path_to_write_dataframe,
                            path_to_label_csv_file=csv_filename)

    if delete_mp3_files:
        print("Cleaning up intermediate mp3 files...")
        for f in glob.glob(path_to_src_files + "*.mp3"):
            os.remove(f)

    if delete_wav_files:
        print("Cleaning up intermediate wav files converted from mp3...")
        for f in glob.glob(path_to_src_files + "*.wav"):
            os.remove(f)

    if delete_split_wav_files:
        print("Cleaning up intermediate split wav files...")
        for f in glob.glob(path_to_split_files + "*.wav"):
            os.remove(f)

    if delete_embeddings:
        print("Cleaning up embeddings...")
        for f in glob.glob(path_to_embeddings + "*.pkl"):
            os.remove(f)

    print("Finished!")

########################################################################
            # Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'Scrape XenoCanto and generate dataframe with labeled examples\
                   that can be used for training/evaluation'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_ARGUMENTS = PARSER._action_groups.pop()
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-b', '--bird_species',
                                    action='store',
                                    help='Input bird species by separating name with \
                                    space and enclosed within quotes, \
                                    for instance "ashy prinia" ',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-o', '--output_path',
                                    action='store',
                                    help='Input new folder path to save downloaed and generated data',
                                    required=True)

    for fileType in ['mp3', 'wav', 'split_wav', 'embeddings']:
        OPTIONAL_ARGUMENTS.add_argument('--delete_'+ fileType +'_files',
                                        dest='delete_' + fileType + '_files',
                                        action='store_true',
                                        help='Delete downloaded ' + fileType + ' files when script finishes')
        OPTIONAL_ARGUMENTS.add_argument('--dont_delete_'+ fileType +'_files',
                                        dest='delete_' + fileType + '_files',
                                        action='store_false',
                                        help='Don\'t delete downloaded ' + fileType + ' files when script finishes')

    # We don't normally delete the mp3 and embeddings files because
    # they usually take the longest to generate. If we were to re-run
    # the script, the time it takes to regenerate these can be saved.
    OPTIONAL_ARGUMENTS.set_defaults(delete_mp3_files=False,
                                    delete_wav_files=True,
                                    delete_split_wav_files=True,
                                    delete_embeddings_files=False)
    PARSER._action_groups.append(OPTIONAL_ARGUMENTS)
    RESULT = PARSER.parse_args()

    xenocanto_to_dataframe(bird_species=RESULT.bird_species,
                           output_path=RESULT.output_path,
                           delete_mp3_files = RESULT.delete_mp3_files,
                           delete_wav_files = RESULT.delete_wav_files,
                           delete_split_wav_files = RESULT.delete_split_wav_files,
                           delete_embeddings = RESULT.delete_embeddings_files)
