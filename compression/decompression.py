import ffmpy
import pandas as pd
import pickle
import numpy as np
import subprocess
import os
import time
import glob

#give the path where the compressed files are stored
compressed_files_path = 'sounds/Compressed_files/compressed_files_libopus/'
#give the path to write the decompressed files
Decompressed_files_path = 'sounds/Decompressed_files/decompressed_files_libopus_wav/'


# Read the balanced data and the get the wav files which are to be compressed and decompressed into list
compressed_files = glob.glob(compressed_files_path+'*.opus')

# create seperte directories if not present to store compressed and decompressed files
if not os.path.exists(Decompressed_files_path):
    os.makedirs(Decompressed_files_path)

# decompressing opus files back into wav format
for file in compressed_files:
    if os.path.exists(Decompressed_files_path+(file.split('/')[-1])[:-4]+'wav'):
        print file.split('/')[-1] + 'has already been decompressed'
    else:
        try:
            subprocess.call('ffmpeg -i ' + file + ' '+ Decompressed_files_path+ (file.split('/')[-1])[:-4]+'wav', shell=True)
            print 'De-compression : ' + file.split('/')[-1] + ' is done..'
        except ValueError:
            n=n+1
            print 'Warning : Skipped ' + n + " as file format is wrong "

print 'De-Compressing the opus files into wav files is Completed..!!'
