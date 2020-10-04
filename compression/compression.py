"""
COmpressing audio files in required format.
AUdio files are taken in balanced numbers
so as to use the decompressed files for training the model
in later stage
"""
import subprocess
import os
import glob
import sys
import argparse



#########################################################################################
            # Description and Help
#########################################################################################

DESCRIPTION = 'Input the path of the original audio files \
              and path to write the compressed audio files'
HELP = 'Supported audio codec formats aac, ac3, mp2, flac, libopus'


#########################################################################################
            # Parsing arguments
#########################################################################################

PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_to_original_audio_files', '--path_to_original_audio_files',
                    action='store', help='Input the path')
PARSER.add_argument('-path_to_compressed_audio_files', '--path_to_compressed_audio_files',
                    action='store', help='Input the path')
PARSER.add_argument('-codec_type', '--codec_type',
                    action='store', help=HELP)
RESULT = PARSER.parse_args()




#########################################################################################
            # set the path where compressed files to be written
#########################################################################################
COMPRESSED_FILES_PATH = RESULT.path_to_compressed_audio_files
ORIGINAL_WAV_FILES_PATH = glob.glob(RESULT.path_to_original_audio_files+"*.wav") + \
                          glob.glob(RESULT.path_to_original_audio_files+"*.WAV")
ORIGINAL_WAV_FILES_LIST = [wav_file.split("/")[-1] for wav_file in ORIGINAL_WAV_FILES_PATH]
TYPE_OF_COMPRESSION = RESULT.codec_type



#########################################################################################
                # create separate directories
#########################################################################################
if not os.path.exists(COMPRESSED_FILES_PATH):
    os.makedirs(COMPRESSED_FILES_PATH)



#########################################################################################
        # Compressing wav files into opus format (It can be changed)
#########################################################################################
if ORIGINAL_WAV_FILES_LIST:
    for ORIGINAL_WAV in ORIGINAL_WAV_FILES_LIST:
        if os.path.exists(COMPRESSED_FILES_PATH + ORIGINAL_WAV[:-3]+TYPE_OF_COMPRESSION):
            print(ORIGINAL_WAV + 'has already been compressed')
        else:
            try:
                subprocess.call('ffmpeg -i '+ RESULT.path_to_original_audio_files + ORIGINAL_WAV +
                                ' -c:a ' + TYPE_OF_COMPRESSION + ' -b:a 64k ' + COMPRESSED_FILES_PATH +
                                ORIGINAL_WAV[:-4]+'.'+TYPE_OF_COMPRESSION, shell=True)
                print('Compression : ' + ORIGINAL_WAV + ' is done..')
            except IOError:
                print('Warning : Skipped ' + ORIGINAL_WAV + " as file doesn't exists in folder")
else:
    print("No Files found in directory")

