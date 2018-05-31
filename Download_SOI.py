import pickle
import pandas as pd
import os
from subprocess import call, check_call, CalledProcessError
import threading
from pydub import AudioSegment
from subprocess import call, check_call, CalledProcessError
import threading

import os
from datetime import datetime
import tensorflow as tf

#Read all the data required
lab = pd.read_csv("data/audioset/class_labels_indices.csv")
un_df = pd.read_csv("data/audioset/unbalanced_train_segments.csv",quotechar='"', skipinitialspace=True, skiprows=2)
print(un_df)

#Specify the  'mid' from class_labels_indices.csv for your required sound of interest( SOI) [eg : '/m/09x0r' in case of 'Speech'] and name it with respective sound [eg : 'Speech' is its corresponding display_name ]
df=pd.DataFrame()
df=un_df.loc[un_df['positive_labels']=='<specify required mid>']
df['labels']='<corresponding display_name_SOI>'
df['YTID']=df['# YTID']

#Download your required number of examples
df=df.iloc[:300][:]
df.index=range(df.shape[0])
print(df.shape)
print(df.head())


def download_clip(YTID, start_seconds, end_seconds):
    url = "https://www.youtube.com/watch?v=" + YTID
    target_file = 'sounds/SOI/' + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)+".wav"

    # No need to download audio file that already has been downloaded
    if os.path.isfile(target_file):
        return

    tmp_filename = 'sounds/tmp_clip_' + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)
    tmp_filename_w_extension = tmp_filename +'.wav'

    try:
        check_call(['youtube-dl', url, '--audio-format', 'wav', '-x', '-o', tmp_filename +'.%(ext)s'])
    except CalledProcessError:
        # do nothing
        print "Exception CalledProcessError!"
        return

    try:
        aud_seg = AudioSegment.from_wav(tmp_filename_w_extension)[start_seconds*1000:end_seconds*1000]

        aud_seg.export(target_file, format="wav")
    except:
        print "Error while reading file!"

    os.remove(tmp_filename_w_extension)



def download_data():


    if not os.path.exists('sounds'):
        os.makedirs('sounds')

    if not os.path.exists('sounds/SOI'):
        os.makedirs('sounds/SOI')

    threads = []

    for index in range(df.shape[0]):
        row = df.iloc[index][:]

        print "Downloading", str(index), "of", df.shape[0], ":", row.YTID

        # download_clip(row.YTID, row.start_seconds, row.end_seconds)
        t = threading.Thread(target=download_clip, args=(row.YTID, row.start_seconds, row.end_seconds))
        threads.append(t)
        t.start()

        if len(threads) > 4:
            threads[0].join()
            threads = threads[1:]

    for t in threads:
        t.join()


if __name__ == "__main__":

    download_data()
