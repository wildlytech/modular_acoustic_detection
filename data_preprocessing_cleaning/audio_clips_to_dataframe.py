
import argparse
import create_base_dataframe
from data_preprocessing_cleaning import mp3_stereo_to_wav_mono
from data_preprocessing_cleaning import split_wav_file
import generating_embeddings
import glob
import os


def audio_clips_to_dataframe(folder_path,
                             delete_mp3_files=False,
                             delete_wav_files=True,
                             delete_split_wav_files=True,
                             delete_embeddings=False):
    """
    Perform entire data preprocessing pipeline to generate dataframe
    for training
    """

    if not folder_path.endswith('/'):
        folder_path += '/'

    path_to_src_files = folder_path
    path_to_split_files = folder_path + "split_wav_files/"
    path_to_embeddings = folder_path + "embeddings/"
    path_to_write_dataframe = folder_path + "dataframe.pkl"

    print("Converting mp3 to wav...")
    mp3_stereo_to_wav_mono.convert_files_directory(path_for_mp3_files=path_to_src_files,
                                                   path_to_save_wavfiles=path_to_src_files)

    print("Splitting wav files into 10 second clips...")
    split_wav_file.audio_split_directory(path_for_wavfiles=path_to_src_files,
                                         path_to_write_chunks=path_to_split_files,
                                         chunk_length_ms=10000)

    print("Generating embeddings for each 10 second clip...")
    generating_embeddings.generate(path_to_write_embeddings=path_to_embeddings,
                                   path_to_wav_files=path_to_split_files)

    print("Building dataframe with features...")
    create_base_dataframe.create_new_dataframe(path_for_saved_embeddings=path_to_embeddings,
                                               path_to_write_dataframe=path_to_write_dataframe)

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

    return path_to_write_dataframe


########################################################################
# Main Function
########################################################################
if __name__ == "__main__":

    DESCRIPTION = 'From folder of audio clips, generate dataframe \
                   with labeled examples that can be used for training/evaluation'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_ARGUMENTS = PARSER._action_groups.pop()
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-f', '--folder_path',
                                    action='store',
                                    help='Input folder path to audio files',
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

    audio_clips_to_dataframe(folder_path=RESULT.folder_path,
                             delete_mp3_files=RESULT.delete_mp3_files,
                             delete_wav_files=RESULT.delete_wav_files,
                             delete_split_wav_files=RESULT.delete_split_wav_files,
                             delete_embeddings=RESULT.delete_embeddings_files)
