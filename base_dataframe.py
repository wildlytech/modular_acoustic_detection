import numpy as np
import pandas as pd
import pickle
import glob
import os
from youtube_audioset import get_csv_data

#Reading a file consiting of the unbalanced data and sound labels
df,labels_binarized=get_csv_data()
labels_csv=pd.read_csv('<path to file class_labels_indices.csv>')
df['positive_labels']=df['positive_labels'].apply(lambda arr: arr.split(','))
df['labels_name']=df['positive_labels'].map(lambda arr:[labels_csv.loc[labels_csv['mid']==x].display_name for x in arr])
df['wav_file'] = df['YTID'].astype(str) + '-' + df['start_seconds'].astype(str) + '-' + df['end_seconds'].astype(str)+'.wav'

#save the dataframe in a pickle file so that it can be reused anytime just by reading it.
with open('unbalanced_data.pkl','w') as f:
    pickle.dump(df,f)
