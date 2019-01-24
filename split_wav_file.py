"""
Function to split a long duration audio file
into chunks of 10second files
"""
import argparse
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks
from colorama import Fore, Style


# Description and help
DESCRIPTION = " Splits the long duration audio files into chunks of 10 seconds"
HELP = "Input the wav file path that is to be splitted"
#parse the input arguments given from command line
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-target_audio_file', '--target_audio_file', action='store',
                    help=HELP)
PARSER.add_argument('-path_to_write_chunks', '--path_to_write_chunks', action='store',
                    help="Input the path to write the splitted chunks of wav files")
RESULT = PARSER.parse_args()


def get_duration_wav_file(wav_file):
    """calculates the wav file duration
    """
    return librosa.get_duration(filename=wav_file)


def start_splitting(chunk_length_ms, wav_file):
    """starts audio splitting
    into chunks of length specified
    """
    file_name = wav_file.split('/')[-1].split(".")[0]
    print "splitting audio files into chunks of 10 seconds :", file_name
    myaudio = AudioSegment.from_wav(wav_file)
    chunks = make_chunks(myaudio, chunk_length_ms)
    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = file_name+"_"+str(i)+'.wav'
        chunk.export(RESULT.path_to_write_chunks+chunk_name, format="wav")
    return len(chunks)


def initiate_audio_split(wav_file, chunk_length_ms):
    """
    make the initiation of process by
    calculating the number of wav files
    """
    # for wav in wav_file:
    total_duration_seconds = get_duration_wav_file(wav_file)
    if total_duration_seconds > 20.0:
        if total_duration_seconds%10 == 0:
            num_wav_files = total_duration_seconds/10
        else:
            num_wav_files = int(total_duration_seconds) - 1
        number_of_chunks_generated = start_splitting(chunk_length_ms, wav_file)
        if num_wav_files == number_of_chunks_generated:
            print Fore.GREEN + "\nTotal number of wav files splitted from" + wav_file.split('/')[-1].split(".")[0] +": ", number_of_chunks_generated
            print Style.RESET_ALL
        else:
            print  Fore.GREEN + "\nTotal number of wav files splitted from" + wav_file.split('/')[-1].split(".")[0] +": ", number_of_chunks_generated
            print Style.RESET_ALL
    else:
        print Fore.RED + "\nWARNING :Audio File must be atleast greater than 20seconds to split"
        print Style.RESET_ALL


if __name__ == "__main__":
    initiate_audio_split(RESULT.target_audio_file, 10000)

