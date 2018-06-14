import numpy
import glob
import pandas as pd
import pickle
from pydub import AudioSegment
from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names,get_csv_data


#specify the path for class_labels_indices.csv file. It consists encoded "mid" as one column and its corresponding "display_name" as other column
labels_csv = pd.read_csv('<path to class_labels_indices.csv>')

#Define the path where the audiofiles(.wav format) are downloaded
all_wav_files = glob.glob('<path to downloaded audio files>/*.wav')
wav_files = []
for i in all_wav_files:
    wav_files.append(i.split('/')[-1])

#calculate the dBFS( decibels to Full scale) to know audio is silent or no.
dbfs=[]
for i in wav_files:
    aud = AudioSegment.from_wav('<path to audio_files>/'+i)
    dbfs.append(aud.dBFS)

#use the get_csv_data() from youtube_audioset.py for reading the unbalanced data
df,labels_binarized = get_csv_data()
df['wav_file'] = df['YTID'].astype(str) +'-'+df['start_seconds'].astype(str)+'-'+df['end_seconds'].astype(str)+'.wav'
df['labels_name'] = df['positive_labels'].map(lambda arr:[labels_csv.loc[labels_csv['mid']=='x'].display_name for x in arr])
#save it in a pickle file
with open('unbalanced_data.pkl','w') as f:
    pickle.dump(df,f)


#create a dataframe with dBFS as column
df_wav_files = pd.DataFrame()
df_wav_files['wav_file'] = wav_files
df_wav_files['dBFS'] = dbfs
req_df = pd.merge(df_wav_files,df,on= 'wav_file', copy=False)
print(req_df.head())


#Filter out the mislabelled ie sound that are labelled as Silence but the audio_clip is not silent and Vice-versa.
#Sounds that are labelled silence, but are not silent. A silent audio will have dBFS value -inf (negative infinite).
df_filtered1 = req_df.loc[req_df['labels'].apply(lambda x: (len(x) == 1)) & req_df.labels.apply(lambda x: 'Silence' in x)]
mislabelled_as_otherthan_silent = df_filtered.loc[df_filtered['dBFS']!=float('-inf')]
list_of_wavfiles = mislabelled_as_otherthan_silent.wav_file.tolist()
#save the balcklisted sounds in a text file
with open('mislabelled_as_silent.txt','w')as f:
    pickle.dump(list_of_wavfiles,f)


#Sounds that are labelled other than Silence but audio clip is pure silent
df_filtered2 = req_df.loc[req['dBFS']==float('-inf')]
mislabelled_as_silence = df_filtered2.loc[df_filtered2['labels_name'].apply(lambda x: 'Silence' not in x)]
list_of_wavfiles2 = mislabelled_as_silence.wav_file.tolist()
#save the blacklisted sounds in a text file
with open('mislabelled_as_other_than_silence.txt','w') as f:
    pickle.dump(list_of_wavfiles2,f)
