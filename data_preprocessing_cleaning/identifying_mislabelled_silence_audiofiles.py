import numpy
import glob
import pandas as pd
import pickle
import argparse
from pydub import AudioSegment




#########################################################################################
               # description and Help
#########################################################################################
DESCRIPTION = '1. Input the path for base dataframe [YTID / wav_file, labels_name] and \n \
               2. Input the path for audio files '
HELP = 'Input the path'



#########################################################################################
            # Arguments and parsing
#########################################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-dataframe_path', '--dataframe_path', action='store',
                    help=HELP)
PARSER.add_argument('-path_for_audio_files', '--path_for_audio_files', action='store',
                    help='Input the path')
RESULT = PARSER.parse_args()



#########################################################################################
                # Helper functions with Docstrings
#########################################################################################
def glob_all_files_from_path(wavfiles_path):
    """
    #Define the path where the audiofiles(.wav format) are downloaded
    """
    all_wav_files_path = glob.glob(wavfiles_path+"*.wav") + glob.glob(wavfiles_path+"*.WAV")
    wav_files = []
    for each_file in all_wav_files_path:
        wav_files.append(each_file.split('/')[-1])
    return all_wav_files_path, wav_files

def get_dbfs(wavfile_path):
    """
    #calculate the dBFS( decibels to Full scale) to know audio is silent or no.
    """
    dbfs = []
    for i in wavfile_path:
        aud = AudioSegment.from_wav(i)
        dbfs.append(aud.dBFS)
    return dbfs

def read_pickle_file(dataframe_path):
    """
    Reads pickle file
    """
    with open(dataframe_path, "rb") as file_obj:
        dataframe = pickle.load(file_obj)
    return dataframe

def create_dataframe(column_list):
    """
    Create Dataframe with given column names
    """
    dataframe = pd.DataFrame(columns=column_list)
    return dataframe


def get_mislabelled_as_silence_files(dataframe):
    """
    Filter out the mislabelled ie sound that are labelled as Silence
    but the audio_clip is not silent and Vice-versa. Sounds that are
    labelled silence, but are not silent. A silent audio will have
    dBFS value -inf (negative infinite).
    """
    dataframe = dataframe.loc[dataframe['labels_name'].apply(lambda x: (len(x) == 1)) & dataframe.labels.apply(lambda x: 'Silence' in x)]
    mislabelled_as_otherthan_silent = dataframe.loc[dataframe['dBFS'] != float('-inf')]
    list_of_wavfiles = mislabelled_as_otherthan_silent.wav_file.tolist()
    #save the balcklisted sounds in a text file
    with open('mislabelled_as_silent.txt', 'w')as file_obj:
        pickle.dump(list_of_wavfiles, file_obj)


def get_mislablled_silence_files(dataframe):
    """
    #Sounds that are labelled other than Silence but audio clip is pure silent
    """
    dataframe = dataframe.loc[dataframe['dBFS'] == float('-inf')]
    mislabelled_as_silence = dataframe.loc[dataframe['labels_name'].apply(lambda x: 'Silence' not in x)]
    list_of_wavfiles = mislabelled_as_silence.wav_file.tolist()
    with open('mislabelled_as_other_than_silence.txt', 'w') as file_obj:
        pickle.dump(list_of_wavfiles, file_obj)




####################################################################################################
            # Main Function
####################################################################################################
if __name__ == "__main__":
    DATAFRAME = read_pickle_file(RESULT.dataframe_path)
    if DATAFRAME["YTID"]:
        DATAFRAME['wav_file'] = DATAFRAME['YTID'].astype(str) +'-'+DATAFRAME['start_seconds'].astype(str)+'-'+DATAFRAME['end_seconds'].astype(str)+'.wav'
        WAVFILES_PATH, WAVFILES_NAMES = glob_all_files_from_path(RESULT.path_for_audio_files)
        DBFS = get_dbfs(WAVFILES_PATH)
        NEW_DATAFRAME = create_dataframe(["wav_file", "dbfs"])
        NEW_DATAFRAME['wav_file'] = WAVFILES_NAMES
        NEW_DATAFRAME['dBFS'] = DBFS
        NEW_DATAFRAME = pd.merge(DATAFRAME, NEW_DATAFRAME, on='wav_file', copy=False)
        get_mislabelled_as_silence_files(NEW_DATAFRAME)
        get_mislablled_silence_files(NEW_DATAFRAME)

    else:
        if DATAFRAME['wav_file']:
            WAVFILES_PATH, WAVFILES_NAMES = glob_all_files_from_path(RESULT.path_for_audio_files)
            DBFS = get_dbfs(WAVFILES_PATH)
            NEW_DATAFRAME = create_dataframe(["wav_file", "dbfs"])
            NEW_DATAFRAME['wav_file'] = WAVFILES_NAMES
            NEW_DATAFRAME['dBFS'] = DBFS
            NEW_DATAFRAME = pd.merge(DATAFRAME, NEW_DATAFRAME, on='wav_file', copy=False)
            get_mislabelled_as_silence_files(NEW_DATAFRAME)
            get_mislablled_silence_files(NEW_DATAFRAME)
        else:
            print("None Found")

