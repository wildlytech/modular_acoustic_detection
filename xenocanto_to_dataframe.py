
import argparse
from data_preprocessing_cleaning import add_xenocanto_labels_to_dataframe
from data_preprocessing_cleaning import audio_clips_to_dataframe
from data_preprocessing_cleaning import remove_rows_from_dataframe
from get_data import xenocanto_scrape


def xenocanto_to_dataframe(bird_species,
                           output_path,
                           delete_mp3_files=False,
                           delete_wav_files=True,
                           delete_split_wav_files=True,
                           delete_embeddings=False):
    """
    Download xeno-canto sound files for particular bird species
    and perform entire data preprocessing pipeline to generate dataframe
    for training
    """

    if not output_path.endswith('/'):
        output_path += '/'

    print("Downloading audio files...")
    path_to_src_files, csv_filename = xenocanto_scrape.scrape(audio_files_path=output_path,
                                                              bird_species=bird_species)

    path_to_dataframe = audio_clips_to_dataframe.audio_clips_to_dataframe(folder_path=path_to_src_files,
                                                                          delete_mp3_files=delete_mp3_files,
                                                                          delete_wav_files=delete_wav_files,
                                                                          delete_split_wav_files=delete_split_wav_files,
                                                                          delete_embeddings=delete_embeddings)

    print("Checking blacklist...")
    remove_rows_from_dataframe.remove_rows_from_dataframe(path_to_dataframe=path_to_dataframe,
                                                          path_to_blacklist_file='blacklisted_audiofiles/xc_blacklist.txt',
                                                          blacklist_item_suffix='.wav',
                                                          id_column='wav_file',
                                                          output_file_name=path_to_dataframe)

    print("Adding labels to dataframe...")
    add_xenocanto_labels_to_dataframe.add_labels_to_dataframe(path_to_feature_dataframe=path_to_dataframe,
                                                              path_to_label_csv_file=csv_filename)

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
                                    help='Input new folder path to save downloaded and generated data',
                                    required=True)

    for fileType in ['mp3', 'wav', 'split_wav', 'embeddings']:
        OPTIONAL_ARGUMENTS.add_argument('--delete_' + fileType + '_files',
                                        dest='delete_' + fileType + '_files',
                                        action='store_true',
                                        help='Delete downloaded ' + fileType + ' files when script finishes')
        OPTIONAL_ARGUMENTS.add_argument('--dont_delete_' + fileType + '_files',
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
                           delete_mp3_files=RESULT.delete_mp3_files,
                           delete_wav_files=RESULT.delete_wav_files,
                           delete_split_wav_files=RESULT.delete_split_wav_files,
                           delete_embeddings=RESULT.delete_embeddings_files)
