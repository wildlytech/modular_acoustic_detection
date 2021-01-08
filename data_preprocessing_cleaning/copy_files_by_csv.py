'''
get wav files list by reading a given csv file to copy wav files from source
to the destination by creating directory with the given csv file name
'''
import glob
import os
import shutil
import pandas as pd
import argparse


def copy_files_by_csv(csv_path, wavfiles_src, wavfiles_dest):
    # read csv to a dataframe
    csv_df = pd.read_csv(csv_path, error_bad_lines=False)
    csv_wav_files_list = csv_df['wav_file'].values.tolist()
    csv_file_name = csv_path.split(".")[-2].split("/")[-1]

    # check to ensure path for the directory
    if not wavfiles_src.endswith('/'):
        wavfiles_src += '/'
    if not wavfiles_dest.endswith('/'):
        wavfiles_dest += '/'

    # create folder if not exists to copy files
    if not os.path.exists(wavfiles_dest + csv_file_name):
        os.makedirs(wavfiles_dest + csv_file_name)
    else:
        pass

    # get all the wav files from the wav files source path
    org_wav_files = glob.glob(wavfiles_src + '*.wav') + glob.glob(wavfiles_src + '*.WAV')
    count = 0
    for wav_file in org_wav_files:
        if wav_file.split("/")[-1] in csv_wav_files_list:
            count += 1
            # copy wav file to the destination path
            shutil.copy(wav_file, wavfiles_dest + csv_file_name)
    print("\nNumber of files copied:", count)

###############################################################################


if __name__ == "__main__":

    DESCRIPTION = 'get wav files list by reading a given csv file to copy wav files from the \
    source to the destination by creating directory with the given csv file name'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    RequiredArguments = PARSER.add_argument_group('required arguments')
    RequiredArguments.add_argument('-csv', '--csv_path', action='store', \
        help='Input csv file path', required=True)
    RequiredArguments.add_argument('-src', '--src_wav_files_dir', action='store', \
        help='Input path to the wav files source', required=True)
    RequiredArguments.add_argument('-dest', '--dest_wav_files_dir', action='store', \
        help='Input path to copy wav files to the destination', required=True)
    RESULT = PARSER.parse_args()

    print("\nGiven CSV file path:", RESULT.csv_path)
    print("\nGiven wav files path:", RESULT.src_wav_files_dir)
    print("\nGiven path to copy wav files:", RESULT.dest_wav_files_dir)

    copy_files_by_csv(RESULT.csv_path, RESULT.src_wav_files_dir, RESULT.dest_wav_files_dir)
