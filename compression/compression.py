"""
COmpressing audio files in required format.
AUdio files are taken in balanced numbers
so as to use the decompressed files for training the model
in later stage
"""
import subprocess
import os
import sys
import argparse
sys.path.insert(0, '../')
import balancing_dataset


# parsing the inputs given
DESCRIPTION = 'Input the path of the original audio files \
              and path to write the compressed audio files'
HELP = 'Supported audio codec formats aac, ac3, mp2, flac, libopus,, mp3'
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_to_original_audio_files', '--path_to_original_audio_files',
                    action='store', help='Input the path')
PARSER.add_argument('-path_to_compressed_audio_files', '--path_to_compressed_audio_files',
                    action='store', help='Input the path')
PARSER.add_argument('-codec_type', '--codec_type',
                    action='store', help=HELP)
RESULT = PARSER.parse_args()

# Read the balanced data and the get the wav files which are to be compressed
REQUIRED_DF = balancing_dataset.balanced_data()
ORIGINAL_WAV_FILE_LIST = REQUIRED_DF['wav_file'].tolist()
TYPE_OF_COMPRESSION = RESULT.codec_type

# set the path where compressed files to be written
COMPRESSED_FILES_PATH = RESULT.path_to_compressed_audio_files
ORIGINAL_WAV_FILES_PATH = RESULT.path_to_original_audio_files

# create seperte directories if not present to store compressed and decompressed files
if not os.path.exists(COMPRESSED_FILES_PATH):
    os.makedirs(COMPRESSED_FILES_PATH)

# Compressing wav files into opus format.
#We have taken opus as an example , you can also change it to mp2,flac and other required format.
for ORIGINAL_WAV in ORIGINAL_WAV_FILE_LIST:
    if os.path.exists(COMPRESSED_FILES_PATH + ORIGINAL_WAV[:-3]+TYPE_OF_COMPRESSION):
        print ORIGINAL_WAV + 'has already been compressed'
    else:
        try:
            print ORIGINAL_WAV
            subprocess.call('ffmpeg -i'+ ORIGINAL_WAV_FILES_PATH + ORIGINAL_WAV +
                            ' -c:a' + TYPE_OF_COMPRESSION + '-b:a 64k' + COMPRESSED_FILES_PATH +
                            ORIGINAL_WAV[:-4]+'.'+TYPE_OF_COMPRESSION, shell=True)
            print 'Compression : ' + ORIGINAL_WAV + ' is done..'
        except IOError:
            print 'Warning : Skipped ' + ORIGINAL_WAV + " as file doesn't exists in folder"

print 'Compressing the wav files into flac files is Completed..!!'
