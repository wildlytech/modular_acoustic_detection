# convert mp3-stereo to wav-mono  -- working
import argparse
from pydub import AudioSegment
import glob
from colorama import Fore


DESCRIPTION = 'Converts mp3-stereo to wav-mono files'
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
RequiredArguments = PARSER.add_argument_group('required arguments')
RequiredArguments.add_argument('-input_mp3_path', action='store', \
	help='Input path to mp3 file or mp3 files directory')
RequiredArguments.add_argument('-path_to_save_wav_files', action='store', \
	help='Input path to save wav file(s)')
RESULT = PARSER.parse_args()


path_for_mp3 = RESULT.input_mp3_path
path_to_save_wavfiles = RESULT.path_to_save_wav_files

def convert_files_directory(path_for_mp3_files):
	# get all the mp3 files in the given path
	mp3_list = glob.glob(path_for_mp3_files+"*.mp3") + glob.glob(path_for_mp3_files+"*.MP3")
	print "Number of mp3 files:", len(mp3_list)
	print "Converting..."
	# loop over the list of mp3 files to convert to wav files
	for mp3_file in mp3_list:
	    mysound = AudioSegment.from_mp3(mp3_file)
	    mono = mysound.set_channels(1)
	    mono.export(path_to_save_wavfiles+mp3_file.split(".")[-2].split("/")[-1]+".wav", format="wav")
	print "Conversion is done !"

def convert_single_file(path_for_mp3_file):
	print "Converting..."
	mysound = AudioSegment.from_mp3(path_for_mp3_file)
	mono = mysound.set_channels(1)
	mono.export(path_to_save_wavfiles+path_for_mp3_file.split(".")\
		[-2].split("/")[-1]+".wav", format="wav")
	print "Conversion is done !"


if __name__ == '__main__':
	if path_for_mp3.split(".")[-1] == 'mp3':
		convert_single_file(path_for_mp3)
	else:
		convert_files_directory(path_for_mp3)
