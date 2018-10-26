"""
Retuns the audiomoth files as a dataframe format
with 'feature' as column and this is furthur used
to predict the sounds using pre-trained model
"""

import pickle
import glob
import pandas as pd
from pydub import AudioSegment


def audio_dataframe(path_to_audio_files):
    """
    Reads the annotated file in .csv file
    """
    labels_csv = pd.read_csv('Nisarg _annotation - Eravikulam.csv')

    # Glob all the audio files ( .WAV) that are of audiomoth recording
    audiomoth_wav_files = glob.glob(path_to_audio_files+'*.WAV')


    #check for audio duration to be 10 seconds. Remove if its not.
    for i in audiomoth_wav_files:
        sound = AudioSegment.from_wav(i)
        if len(sound) != 10000:
            print i + 'is removed, as it is not a ten second audio'
            audiomoth_wav_files.remove(i)

    # create lists that has audiomoth file name and id's
    audiomoth_id = []
    wave_files = []
    for i in audiomoth_wav_files:
        audiomoth_id.append((i.split('/')[-1])[:-4])
        wave_files.append(i.split('/')[-1])
    print 'first element :', audiomoth_id[0]

    # create a dataframe
    data_frame = pd.DataFrame()
    data_frame['audiomoth_id'] = audiomoth_id
    data_frame['wav_file'] = wave_files


    # Merge the dataframe to get the required dataframe
    req = pd.merge(labels_csv, data_frame, on='wav_file')
    data_frame = req

    return data_frame


def embeddings_on_dataframe(path_to_audio_files, path_to_embeddings):
    """
    Returns the dataframe with 'feature' as column of those
    that are annotated
    """

    data_frame = audio_dataframe(path_to_audio_files)

    # Get all the pickle files into list
    pickle_files = glob.glob(path_to_embeddings+'*.pkl')

    # create a list of audiomoth file name and its id's
    audiomoth_id = []
    for i in pickle_files:
        audiomoth_id.append((i.split('/')[-1]) [:-4])

    #Read all the embeddings
    clf1_test = []

    for i in pickle_files:
        try:
            with open(i, 'rb') as file_obj:
                arb_wav = pickle.load(file_obj)
            clf1_test.append(arb_wav)
        except:
            print 'Test Pickling Error'

    # create a dataframe with embeddings as column
    emb_df = pd.DataFrame()
    emb_df['audiomoth_id'] = audiomoth_id
    emb_df['features'] = clf1_test
    final = pd.merge(data_frame, emb_df, on='audiomoth_id')

    return final

def embeddings_without_annotation(path_to_embeddings):
    """
    Returns dataframe with 'feature' as column as those
    that are not annotated
    """

    pickle_files = glob.glob(path_to_embeddings+'*.pkl')

    audiomoth_id = []
    for i in pickle_files:
        audiomoth_id.append((i.split('/')[-1])[:-4])

    clf1_test = []

    for i in pickle_files:
        try:
            with open(i, 'rb') as file_obj:
                arb_wav = pickle.load(file_obj)
            clf1_test.append(arb_wav)
        except:
            print 'pickle read ERROR'

    #create a dataframe with embeddings as column
    emb_df = pd.DataFrame()
    emb_df['audiomoth_id'] = audiomoth_id
    emb_df['features'] = clf1_test

    #print the head and shape of the datframe
    print emb_df.head()
    print 'shape :', emb_df.shape
    return emb_df
