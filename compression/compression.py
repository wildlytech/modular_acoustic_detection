import ffmpy
import pandas as pd
import pickle
import numpy as np
import subprocess
import os
import time



# Read the balanced data and the get the wav files which are to be compressed
req = balancing_dataset.balanced_data()
original_wav_file_list = req['wav_file'].tolist()

# set the path where compressed files to be written
compressed_file_path = 'sounds/Compressed_files/compressed_files_libopus/'
audio_files_path = 'sounds/speech/'

# create seperte directories if not present to store compressed and decompressed files
if not os.path.exists(compressed_file_path):
    os.makedirs(compressed_file_path)

# Compressing wav files into opus format. We have taken opus as an example , you can also change it to mp2,flac and other required format.
for original_wav in original_wav_file_list:
    if os.path.exists(compressed_file_path + original_wav[:-3]+'opus'):
        print original_wav + 'has already been compressed'
    else:
        try:
            print original_wav
            subprocess.call('ffmpeg -i'+ audio_files_path + original_wav + ' -c:a libopus -b:a 64k' + compressed_file_path + original_wav[:-4]+'.opus', shell=True)
            print 'Compression : ' + original_wav + ' is done..'
        except IOError:
            print 'Warning : Skipped ' + original_wav + " as file doesn't exists in folder"

print 'Compressing the wav files into flac files is Completed..!!'
