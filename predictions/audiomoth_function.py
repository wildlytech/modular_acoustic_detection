import pickle
import glob
sys.path.insert(0, '../')
import youtube_audioset
import pandas as pd
import numpy as np
import pydub
from pydub import AudioSegment


#for reading all the audiomoth (.WAV ) files and creating a dataframe
def audio_dataframe(path_to_audio_files):

    #Read the annotated file
    labels_csv = pd.read_csv('Nisarg _annotation - Eravikulam.csv')

    # Glob all the audio files ( .WAV) that are of audiomoth recording
    audiomoth_wav_files = glob.glob(path_to_audio_files+'*.WAV')


    #check for audio duration to be 10 seconds. Remove if its not.
    for i in audiomoth_wav_files:
        sound = AudioSegment.from_wav(i)
        if not len(sound)==10000:
            print i + 'is removed, as it is not a ten second audio'
            audiomoth_wav_files.remove(i)

    # create lists that has audiomoth file name and id's
    audiomoth_id = []
    wave_files= []
    for i in audiomoth_wav_files:
        audiomoth_id.append((i.split('/')[-1])[:-4])
        wave_files.append(i.split('/')[-1])
    print 'first element :', audiomoth_id[0]

    # create a dataframe
    df = pd.DataFrame()
    df['audiomoth_id'] = audiomoth_id
    df['wav_file'] = wave_files


    # Merge the dataframe to get the required dataframe
    req = pd.merge(labels_csv,df,on='wav_file')
    df=req

    return df

#reading all the embeddings
def embeddings_on_dataframe(path_to_audio_files,path_to_embeddings):

    df = audio_dataframe(path_to_audio_files, path_to_embeddings)

    # Get all the pickle files into list 
    pickle_files = glob.glob(path_to_embeddings+'*.pkl')

    # create a list of audiomoth file name and its id's
    audiomoth_id=[]
    for i in pickle_files:
        audiomoth_id.append( ( i.split('/')[-1]) [:-4])

    #Read all the embeddings
    clf1_test=[]

    for i in pickle_files:
        try:
            with open(i,'rb') as f:
                arb_wav = pickle.load(f)
            clf1_test.append(arb_wav)
        except :
            print 'Test Pickling Error'

    # create a dataframe with embeddings as column
    emb_df= pd.DataFrame()
    emb_df['audiomoth_id'] = audiomoth_id
    emb_df['features'] = clf1_test
    final = pd.merge(df,emb_df,on='audiomoth_id')


    return final

    
