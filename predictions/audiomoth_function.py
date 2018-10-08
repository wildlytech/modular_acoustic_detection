import pickle
import glob
import youtube_audioset
import pandas as pd
import numpy as np
import pydub
from pydub import AudioSegment



def audio_dataframe():

    #Read the annotated file
    labels_csv = pd.read_csv('Nisarg _annotation - Eravikulam.csv')
    print labels_csv.head()


    # Glob all the audio files ( .WAV) that are of audiomoth recording
    audiomoth_wav_files = glob.glob('/media/wildly/1TB-HDD/AudioMoth/test_wave/*.WAV')


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
        audiomoth_id.append((i.split('/')[-1])[:8])
        wave_files.append(i.split('/')[-1])
    print 'first element :', audiomoth_id[0]

    # create a dataframe
    df = pd.DataFrame()
    df['audiomoth_id'] = audiomoth_id
    df['wav_file'] = wave_files

    # Give manually start and end seconds as they are only 10seconds audio  files
    df['start_seconds'] = 0.0
    df['end_seconds'] = 10.0
    df['labels_name'] = ['Wind'] * df.shape[0]
    df['wav_file_order'] = df['audiomoth_id'].astype(str) + '-' + df['start_seconds'].astype(str) + '-' + df['end_seconds'].astype(str) + '.wav'

    # Merge the dataframe to get the required dataframe
    req = pd.merge(labels_csv,df,on='wav_file')
    df=req
    print df.head()

    return df

def embeddings_on_dataframe():
    df = audio_dataframe()

    # Get all the pickle files into list 
    pickle_files = glob.glob('/media/wildly/1TB-HDD/goertzel_data_ervikulam/*.pkl')

    # create a list of audiomoth file name and its id's
    audiomoth_id=[]
    for i in pickle_files:
        audiomoth_id.append( ( i.split('/')[-1]) [:10] + '.WAV')

    #Read all the embeddings
    clf1_test=[]

    for i in pickle_files:
        try:
            with open(i,'rb') as f:
                arb_wav = pickle.load(f)
            clf1_test.append(arb_wav)
        except :
            print 'Test Pickling Error'

    print np.array(clf1_test).shape
    clf1_test = np.array(clf1_test).reshape(-1,1280,1)

    # create a dataframe with embeddings as column
    emb_df= pd.DataFrame()
    emb_df['wav_file'] = audiomoth_id
    final = pd.merge(df,emb_df,on='wav_file')
    final = final.drop(columns=['labels_name','start_seconds','end_seconds','wav_file_order'])
    print final.head()

    return clf1_test, final

    
