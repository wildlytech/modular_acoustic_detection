"""
Function to split a long duration audio file
into chunks of 10second files
"""
import argparse
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks
from colorama import Fore, Style
import glob
import os

###############################################################################
                    # Helper Function
###############################################################################

def get_duration_wav_file(wav_file):
    """
    Calculates the wav file duration
    """
    return librosa.get_duration(filename=wav_file)


def start_splitting(chunk_length_ms, wav_file, path_to_write_chunks):
    """
    Starts audio splitting into chunks of length specified
    """

    # append a slash at the end of the path to save wav files
    # if it is not there already
    if not path_to_write_chunks.endswith('/'):
        path_to_write_chunks += '/'

    # If it doesn't exist, create directory for output file
    if not os.path.exists(path_to_write_chunks):
        os.makedirs(path_to_write_chunks)

    file_name = wav_file.split('/')[-1].split(".")[0]
    print("splitting audio files into chunks of", chunk_length_ms / 1000.0, "seconds :", file_name)
    myaudio = AudioSegment.from_wav(wav_file)
    chunks = make_chunks(myaudio, chunk_length_ms)
    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = file_name+"_"+str(i)+'.wav'

        # if the last chunk is not of length, then pad it with silence
        if len(chunk) < chunk_length_ms:
            print("Padding last file with", (chunk_length_ms - len(chunk)) / 1000.0, "seconds of silence")
            chunk = chunk + AudioSegment.silent(duration = chunk_length_ms - len(chunk),
                                                frame_rate=chunk.frame_rate)

        chunk.export(path_to_write_chunks+chunk_name, format="wav")
    return len(chunks)


def audio_split_single_file(wav_file, path_to_write_chunks, chunk_length_ms):
    """
    Make the initiation of process by
    calculating the number of wav files
    """
    # Get length in seconds
    chunk_length_sec = chunk_length_ms / 1000.0

    # for wav in wav_file:
    total_duration_seconds = get_duration_wav_file(wav_file)

    if total_duration_seconds <= chunk_length_sec:
        print(Fore.RED + "\nWARNING :Audio File must be at least greater than", chunk_length_sec, "seconds to split")
        print(Style.RESET_ALL)

    num_wav_files = (total_duration_seconds // chunk_length_sec) + ((total_duration_seconds % chunk_length_sec) == 0)

    number_of_chunks_generated = start_splitting(chunk_length_ms=chunk_length_ms,
                                                 wav_file=wav_file,
                                                 path_to_write_chunks=path_to_write_chunks)
    if num_wav_files == number_of_chunks_generated:
        print(Fore.GREEN + "\nTotal number of wav files splitted from", wav_file.split('/')[-1].split(".")[0], ":", number_of_chunks_generated)
        print(Style.RESET_ALL)
    else:
        print(Fore.GREEN + "\nTotal number of wav files splitted from", wav_file.split('/')[-1].split(".")[0], ":", number_of_chunks_generated)
        print(Style.RESET_ALL)


def audio_split_directory(path_for_wavfiles, path_to_write_chunks, chunk_length_ms):
    """
    Make the initiation of process by
    calculating the number of wav files
    """
    # get all the wav files in the given directory

    # append a slash at the end of the path to read wav files
    # if it is not there already
    if not path_for_wavfiles.endswith('/'):
        path_for_wavfiles += '/'

    wav_files_list = glob.glob(path_for_wavfiles+"*.wav") + glob.glob(path_for_wavfiles+"*.WAV")
    # iterate all the wav files in the list to split
    for wav_file in wav_files_list:
        audio_split_single_file(wav_file=wav_file,
                                path_to_write_chunks=path_to_write_chunks,
                                chunk_length_ms=chunk_length_ms)


###############################################################################
            # Main Function
###############################################################################
if __name__ == "__main__":

    ###########################################################################
            # Description and help
    ###########################################################################

    DESCRIPTION = " Splits the long duration audio files into chunks of 10 seconds"
    HELP = "Input path for wav files directory or input single wav file path to split into chunks of 10seconds"


    ###########################################################################
                # Parse the arguments
    ###########################################################################
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-path_for_wavfiles', '--path_for_wavfiles', action='store',
                                    help=HELP,
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-path_to_write_chunks', '--path_to_write_chunks', action='store',
                                    help="Input the path to write the splitted chunks of wav files",
                                    required=True)
    RESULT = PARSER.parse_args()

    path_for_wavfiles = RESULT.path_for_wavfiles
    path_to_write_chunks = RESULT.path_to_write_chunks

    if path_for_wavfiles.split(".")[-1].lower() == 'wav':
        audio_split_single_file(wav_file = path_for_wavfiles,
                                path_to_write_chunks = path_to_write_chunks,
                                chunk_length_ms = 10000)
    else:
        audio_split_directory(path_for_wavfiles = path_for_wavfiles,
                              path_to_write_chunks = path_to_write_chunks,
                              chunk_length_ms = 10000)
