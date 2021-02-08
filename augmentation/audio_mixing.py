import sys
import os
import argparse
import pickle
import pandas as pd
from scipy.io import wavfile
from pydub import AudioSegment
import glob
PATH_FOR_DATA = 'diff_class_datasets/Datasets/'


###############################################################################
# description and Help
###############################################################################

DESCRIPTION = 'Input the type of youtube sounds to mix. It can be motor, explosion, human, nature \
                domestic, tools.'
HELP = 'Input type of audio sound'


###############################################################################
# Arguments and parsing
###############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-use_youtube_sounds', '--use_youtube_sounds', action='store',
                    help=HELP)
PARSER.add_argument('-type_one', '--type_one', action='store',
                    help=HELP)
PARSER.add_argument('-type_two', '--type_two', action='store',
                    help=HELP)
PARSER.add_argument('-path_to_save_mixed_sounds', '--path_to_save_mixed_sounds', action='store',
                    help='Input the path')
PARSER.add_argument('-path_type_one_audio_files', '--path_type_one_audio_files', action='store',
                    help='Input the path')
PARSER.add_argument('-path_type_two_audio_files', '--path_type_two_audio_files', action='store',
                    help='Input the path')
RESULT = PARSER.parse_args()


###############################################################################
# Define Constants
###############################################################################

SAMPLING_RATE_ONE = 48000
SAMPLING_RATE_TWO = 44100
MAXIMUM_LIMIT_EXAMPLES = 8000


###############################################################################
# Function to read all data frames from Youtube data
###############################################################################
def balanced_data():
    """
    Function to read all data frames and balancing
    """

    # Files with single class
    with open(PATH_FOR_DATA + 'pure/Explosion/pure_exp_7957.pkl', 'rb') as file_obj:
        pure_exp = pickle.load(file_obj)
    with open(PATH_FOR_DATA + 'pure/Motor/pure_mot_76045.pkl', 'rb') as file_obj:
        pure_mot = pickle.load(file_obj)
    with open(PATH_FOR_DATA + 'pure/Human_sounds/pure_hum_46525.pkl', 'rb') as file_obj:
        pure_hum = pickle.load(file_obj)
    # with open(PATH_FOR_DATA+'pure/Wood/pure_wod_1115.pkl', 'rb') as file_obj:
    #     pure_wod = pickle.load(file_obj)
    with open(PATH_FOR_DATA + 'pure/Nature_sounds/pure_nat_13527.pkl', 'rb') as file_obj:
        pure_nat = pickle.load(file_obj)
    with open(PATH_FOR_DATA + 'pure/Domestic/pure_dom_9497.pkl', 'rb') as file_obj:
        pure_dom = pickle.load(file_obj)
    with open(PATH_FOR_DATA + 'pure/Tools/pure_tools_8113.pkl', 'rb') as file_obj:
        pure_tools = pickle.load(file_obj)
    # with open(PATH_FOR_DATA+'pure/Wild/pure_wild_7061.pkl','rb') as file_obj:
    #     pure_wild=pickle.load(file_obj)

    # Balancing and experimenting
    exp = pd.concat([pure_exp], ignore_index=True)
    mot = pd.concat([pure_mot], ignore_index=True)
    hum = pd.concat([pure_hum], ignore_index=True)
    # wood= pd.concat([pure_wod],ignore_index=True)
    nat = pd.concat([pure_nat], ignore_index=True)
    dom = pd.concat([pure_dom], ignore_index=True)
    tools = pd.concat([pure_tools], ignore_index=True)

    return mot, exp, hum, nat, dom, tools


###############################################################################
# Getting wavfiles from different paths.
# Change paths according (Add or delete
# if there are multiple paths where audio
# files are stored)
###############################################################################
def get_file_path(filename):
    if os.path.exists("/media/wildly/Seagate/Audio_files/unbal_audio_data_178k/" + filename):
        return "/media/wildly/Seagate/Audio_files/unbal_audio_data_178k/" + filename
    elif os.path.exists("/media/wildly/Seagate/Audio_files/priority1/" + filename):
        return "/media/wildly/Seagate/Audio_files/priority1/" + filename
    elif os.path.exists("/media/wildly/Seagate/Audio_files/priority_2/" + filename):
        return "/media/wildly/Seagate/Audio_files/priority_2/" + filename
    else:
        return None


###############################################################################
# Get different sampling rates audio Files in separate lists (Youtube)
###############################################################################
def get_samplerate_44_48(dataframe):
    samp_44k = []
    samp_48k = []
    for each in dataframe['wav_file'].values.tolist():
        path_ = get_file_path(each)
        if path_:
            sample_rate, _ = wavfile.read(path_)
            if sample_rate == SAMPLING_RATE_TWO:
                samp_44k.append(each)
            elif sample_rate == SAMPLING_RATE_ONE:
                samp_48k.append(each)
            else:
                pass
        else:
            pass
    return samp_44k, samp_48k


###############################################################################
# Get different sampling rates audio Files in separate lists (When paths are given)
###############################################################################
def get_samplerate_44_48_paths(wavfiles_list):
    print("Sorting Files")
    samp_44k = []
    samp_48k = []
    for each in wavfiles_list:
        sample_rate, _ = wavfile.read(each)
        if sample_rate == SAMPLING_RATE_TWO:
            samp_44k.append(each)
        elif sample_rate == SAMPLING_RATE_ONE:
            samp_48k.append(each)
        else:
            pass
    return samp_44k, samp_48k


###############################################################################
# Returning the selection type
###############################################################################
def get_selected_type(type_sound):
    if type_sound == "motor":
        return '0'
    elif type_sound == "explosion":
        return '1'
    elif type_sound == "human":
        return '2'
    elif type_sound == "nature":
        return '3'
    elif type_sound == "domestic":
        return '4'
    elif type_sound == "tools":
        return '5'
    else:
        return None


###############################################################################
# Mixing audio clips (Youtube)
###############################################################################
def start_mixing_two_audioclips(wavfile_list1, wavfile_list2, min_examples):
    print("started mixing audio clips")
    for each_file in zip(wavfile_list1[:min_examples], wavfile_list2[:min_examples]):
        if os.path.exists(RESULT.path_to_save_mixed_sounds + each_file[0][:-4] + "__mix__" + each_file[1][:-4] + ".wav"):
            pass
        else:
            path = get_file_path(each_file[0])
            path2 = get_file_path(each_file[1])
            sound1 = AudioSegment.from_wav(path)
            sound2 = AudioSegment.from_wav(path2)
            output = sound1.overlay(sound2)
            output.export(RESULT.path_to_save_mixed_sounds + each_file[0][:-4] + "__mix__" + each_file[1][:-4] + ".wav", format="wav")


###############################################################################
# Mixing audio clips from paths given
###############################################################################
def start_mixing_two_audioclips_from_path(wavfile_list1, wavfile_list2, min_examples):
    print("started mixing")
    for each_file in zip(wavfile_list1[:min_examples], wavfile_list2[:min_examples]):
        if os.path.exists(RESULT.path_to_save_mixed_sounds + each_file[0].split("/")[-1][:-4] + "__mix__" + each_file[1].split("/")[-1][:-4] + ".wav"):
            pass
        else:
            sound1 = AudioSegment.from_wav(each_file[0])
            sound2 = AudioSegment.from_wav(each_file[1])
            output = sound1.overlay(sound2)
            output.export(RESULT.path_to_save_mixed_sounds + each_file[0].split("/")[-1][:-4] + "__mix__" + each_file[1].split("/")[-1][:-4] + ".wav", format="wav")


###############################################################################
# Main Function
###############################################################################
if __name__ == "__main__":

    if RESULT.use_youtube_sounds:
        DATAFRAME = balanced_data()
        if RESULT.type_one:
            TYPEONE_INDEX = get_selected_type(RESULT.type_one)
            if TYPEONE_INDEX:
                TYPEONE44K, TYPEONE48K = get_samplerate_44_48(DATAFRAME[int(TYPEONE_INDEX)][:MAXIMUM_LIMIT_EXAMPLES])
            else:
                print("Invalid type one sound Selected")
                sys.exit(1)
        else:
            print("Select type of sounds to mix")
            sys.exit(1)
        if RESULT.type_two:
            TYPETWO_INDEX = get_selected_type(RESULT.type_two)
            if TYPETWO_INDEX:
                TYPETWO44K, TYPETWO48K = get_samplerate_44_48(DATAFRAME[int(TYPETWO_INDEX)][:MAXIMUM_LIMIT_EXAMPLES])
            else:
                print("Invalid type two sound Selected")
                sys.exit(1)
        else:
            print("Select type of sounds to mix")
            sys.exit(1)

        MIN_44K = min(len(TYPEONE44K), len(TYPETWO44K))
        start_mixing_two_audioclips(TYPEONE44K, TYPETWO44K, MIN_44K)

        MIN_48K = min(len(TYPEONE48K), len(TYPETWO48K))
        start_mixing_two_audioclips(TYPEONE48K, TYPETWO48K, MIN_48K)
    else:
        if RESULT.path_type_one_audio_files:
            TYPEONE_FILES_PATH = glob.glob(RESULT.path_type_one_audio_files + "*.wav") + \
                glob.glob(RESULT.path_type_one_audio_files + "*.WAV")
        else:
            print("Input path for type of sound")
        if RESULT.path_type_two_audio_files:
            TYPETWO_FILES_PATH = glob.glob(RESULT.path_type_two_audio_files + "*.wav") + \
                glob.glob(RESULT.path_type_two_audio_files + "*.WAV")
        else:
            print("Input path for type of sound")

        TYPEONE44K, TYPEONE48K = get_samplerate_44_48_paths(TYPEONE_FILES_PATH)
        TYPETWO44K, TYPETWO48K = get_samplerate_44_48_paths(TYPETWO_FILES_PATH)

        MIN_44K = min(len(TYPEONE44K), len(TYPETWO44K))
        start_mixing_two_audioclips_from_path(TYPEONE44K, TYPETWO44K, MIN_44K)

        MIN_48K = min(len(TYPEONE48K), len(TYPETWO48K))
        start_mixing_two_audioclips_from_path(TYPEONE48K, TYPETWO48K, MIN_48K)
