# convert mp3-stereo to wav-mono  -- working
import argparse
from pydub import AudioSegment
import glob
import os


def convert_files_directory(path_for_mp3_files, path_to_save_wavfiles):
    """
    Convert all mp3 files in a particular directory to wav files
    """

    # append a slash at the end of the path to read mp3 files
    # if it is not there already
    if not path_for_mp3_files.endswith('/'):
        path_for_mp3_files += '/'

    # append a slash at the end of the path to save wav files
    # if it is not there already
    if not path_to_save_wavfiles.endswith('/'):
        path_to_save_wavfiles += '/'

    # If it doesn't exist, create directory for output file
    if not os.path.exists(path_to_save_wavfiles):
        os.makedirs(path_to_save_wavfiles)

    # get all the mp3 files in the given path
    mp3_list = glob.glob(path_for_mp3_files + "*.mp3") + glob.glob(path_for_mp3_files + "*.MP3")
    print("Number of mp3 files:", len(mp3_list))
    print("Converting...")
    # loop over the list of mp3 files to convert to wav files
    for mp3_file in mp3_list:
        mysound = AudioSegment.from_mp3(mp3_file)
        mono = mysound.set_channels(1)
        mono.export(path_to_save_wavfiles + mp3_file.split(".")[-2].split("/")[-1] + ".wav", format="wav")
    print("Conversion is done !")


def convert_single_file(path_for_mp3_file, path_to_save_wavfiles):
    """
    Convert single mp3 file to a wav file
    """

    print("Converting...")

    # append a slash at the end of the path to save wav files
    # if it is not there already
    if not path_to_save_wavfiles.endswith('/'):
        path_to_save_wavfiles += '/'

    # If it doesn't exist, create directory for output file
    if not os.path.exists(path_to_save_wavfiles):
        os.makedirs(path_to_save_wavfiles)

    mysound = AudioSegment.from_mp3(path_for_mp3_file)
    mono = mysound.set_channels(1)
    mono.export(path_to_save_wavfiles + \
                path_for_mp3_file.split(".")[-2].split("/")[-1] + ".wav",
                format="wav")
    print("Conversion is done !")


if __name__ == '__main__':

    DESCRIPTION = 'Converts mp3-stereo to wav-mono files'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-input_mp3_path', action='store', \
        help='Input path to mp3 file or mp3 files directory',
        required=True)
    REQUIRED_ARGUMENTS.add_argument('-path_to_save_wav_files', action='store', \
        help='Input path to save wav file(s)',
        required=True)
    RESULT = PARSER.parse_args()

    path_for_mp3 = RESULT.input_mp3_path
    path_to_save_wavfiles = RESULT.path_to_save_wav_files

    if path_for_mp3.split(".")[-1].lower() == 'mp3':
        convert_single_file(path_for_mp3, path_to_save_wavfiles)
    else:
        convert_files_directory(path_for_mp3, path_to_save_wavfiles)
