import pickle
import glob
import youtube_audioset
import pandas as pd
import numpy as np
import pydub
from pydub import AudioSegment


def audio_dataframe(audio_files_path):

    #Read the annotated file
    labels_csv = pd.read_csv('Nisarg _annotation - Eravikulam.csv')
    print labels_csv.head()

    # Glob all the audio files ( .WAV) that are of audiomoth recording
    audiomoth_wav_files = glob.glob(audio_files_path + '*.WAV')


    #check for audio duration to be 10 seconds. Remove if its not.
    for each_wav in audiomoth_wav_files:
        sound = AudioSegment.from_wav(each_wav)
        if not len(sound)==10000:
            print each_wav + 'is removed, as it is not a ten second audio'
            audiomoth_wav_files.remove(i)

    # create lists that has audiomoth file name and id's
    audiomoth_id = []
    wave_files= []
    for each_wav in audiomoth_wav_files:
        audiomoth_id.append((each_wav.split('/')[-1])[:8])
        wave_files.append(each_wav.split('/')[-1])
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

    #return dataframe
    return df


def dataframe_with_frequency_components(audio_files_path,frequency_component_files):

    #get the dataframe of the audiomoth files (.WAV files )
    df = audio_dataframe(audio_files_path)

    # Get all the pickle files into list
    pickle_files = glob.glob(frequency_component_files + '*.pkl')
    audiomoth_id=[]
    for pkl in pickle_files:
        audiomoth_id.append( ( pkl.split('/')[-1]) [:10] + '.WAV')

    #Read all files in  frequency components  folder to a list
    clf1_test=[]

    for each_file in pickle_files:
        try:
            with open(each_file,'rb') as f:
                arb_wav = pickle.load(f)

            #stacj the frequency components at the 3rd axis
            clf1_test.append(np.dstack( (arb_wav[0].reshape((10,8000)) ,arb_wav[1].reshape((10,8000)), arb_wav[2].reshape((10,8000)), arb_wav[3].reshape((10,8000)) ) ))
        except :
            print 'Test Pickling Error'
    #print the shape of the array.
    print np.array(clf1_test).shape

    # conver the list into numpy array with reshaping it, so as to be same as input to the goertzel model
    clf1_test = np.array(clf1_test).reshape((-1,10,8000,4))

    #normalization of the data
    clf1_test = clf1_test / np.linalg.norm(clf1_test)

    # create a dataframe and returning it along with the frequency components data
    emb_df= pd.DataFrame()
    emb_df['wav_file'] = audiomoth_id
    final = pd.merge(df,emb_df,on='wav_file')
    final = final.drop(columns=['labels_name','start_seconds','end_seconds','wav_file_order'])

    return clf1_test, final
