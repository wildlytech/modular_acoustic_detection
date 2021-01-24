import os
import time
import csv
import argparse
from ftplib import FTP
import pandas as pd
from pydub import AudioSegment
from predictions.binary_relevance_model import predict_on_wavfile_binary_relevance
import sys

global FTP_OBJ

############################################################################
# List only wav files into
############################################################################
def check_for_wav_only(list_values):
    """
    Get the list of wav files .wav and .WAV format only
    """
    wav_files = []
    for each_value in list_values:
        if each_value[-3:] == "WAV" or each_value[-3:] == "wav":
            wav_files.append(each_value)
    print("Total file: ", len(wav_files))
    return wav_files


############################################################################
# Sort files based on time
############################################################################
def sorting_files_same_as_upload_order(wav_files_list):
    """
    sorting files based on the timestamp i.e upload time
    """
    DICT = {}
    count = 0
    print("Files are being sorted..")
    for name in wav_files_list:
        if (name[-3:] == 'wav') or (name[-3:] == 'WAV'):
            # returns last uploaded file's time with utc
            time1 = FTP_OBJ.voidcmd("MDTM " + name)
            count += 1
            DICT[name] = time1[4:]
    sorted_list = sorted((value, key) for (key, value) in list(DICT.items()))
    sorted_filenames = [element[0] for element in sorted_list]

    return sorted_filenames


############################################################################
# Connect ftp server
############################################################################
def call_for_ftp(ftp_folder_path, ftp_password):
    """
    Connect to FTP and display all the wav files present in directory
    """
    global FTP_OBJ

    FTP_OBJ = FTP('34.211.117.196', user='user-u0xzU', passwd=ftp_password)
    print("Connection Status : connected to FTP")

    if not ftp_folder_path.endswith('/'):
        ftp_folder_path += '/'

    FTP_OBJ.cwd(ftp_folder_path)
    ex = FTP_OBJ.nlst()
    wav_files_only = check_for_wav_only(ex)

    # If files are needed in the order in which they are uploaded  the uncomment the below the line
    # This will take a quite a bit of time if there are more number of files

    # wav_files_only = sorting_files_same_as_upload_order(wav_files_only)

    dataframe = pd.DataFrame()
    dataframe["FileNames"] = wav_files_only
    dataframe = dataframe.sort_values(["FileNames"], ascending=[1])
    return dataframe

def start_batch_run(predictions_cfg_json,
                    path_for_folder,
                    wav_files_list,
                    csv_filename,
                    online):
    """
    Writes the predicted results on to csvfile row wise
    """
    malformed = []

    # Check if the csv file is already existing or not.
    # If it is existing then append the result to same csv file
    file_exists = os.path.exists(csv_filename)

    if file_exists and os.stat(csv_filename).st_size > 0:
        # File exists and has at least a header row
        data_read = pd.read_csv(csv_filename)
        list_of_files_predicted = data_read['FileNames'].tolist()
    else:
        list_of_files_predicted = []

    with open(csv_filename, "a" if file_exists else "w") as file_object:
        wav_information_object = csv.writer(file_object)

        for index, each_file in enumerate(wav_files_list):
            if each_file not in list_of_files_predicted:

                # Predict the result and save the result to the csv file
                print("predicting for :", each_file)

                try:
                    path = path_for_folder + each_file

                    if online:
                        if os.path.exists(path):
                            print("path Exists")
                        else:
                            with open(path, 'wb') as file_obj:
                                global FTP_OBJ
                                FTP_OBJ.retrbinary('RETR ' + each_file, file_obj.write)

                    pred_df = predict_on_wavfile_binary_relevance.main(predictions_cfg_json=predictions_cfg_json,
                                                                       path_for_wavfile=path,
                                                                       print_results=False)

                    pred_df['FileNames'] = each_file
                    pred_df['dBFS'] = AudioSegment.from_wav(path).dBFS

                    if not file_exists:
                        wav_information_object.writerow(pred_df.columns.to_list())
                        file_object.flush()
                        # Set file exists to false to prevent writing header again
                        file_exists = True

                    wav_information_object.writerow(pred_df.iloc[0].tolist())
                    file_object.flush()

                except ValueError:
                    print("malformed index", index)
                    malformed.append(path)

    return malformed

def start_batch_run_ftp_live(ftp_folder_path,
                             ftp_password,
                             download_folder_path,
                             predictions_cfg_json,
                             csv_filename):

    if not download_folder_path.endswith('/'):
        download_folder_path += '/'

    if not os.path.exists(download_folder_path):
        os.makedirs(download_folder_path)

    inter_df = call_for_ftp(ftp_folder_path, ftp_password)
    wav_files_list = inter_df['FileNames'].tolist()

    malformed = start_batch_run(predictions_cfg_json=predictions_cfg_json,
                                path_for_folder=download_folder_path,
                                wav_files_list=wav_files_list,
                                csv_filename=csv_filename,
                                online=True)

    for path in malformed:
        os.remove(path)

############################################################################
            # Main Function
############################################################################
if __name__ == "__main__":

    ########################################################################
    # Target Path or Folder in FTP
    ########################################################################
    DESCRIPTION = "Generates the csv file with prediction results"
    HELP = "Give the Required Arguments"


    ########################################################################
    # Target Path or Folder in FTP
    ########################################################################
    DEFAULT_CSV_FILENAME = "predictions_FTP_folder.csv"


    ########################################################################
    # Parsing argument
    ########################################################################
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()

    REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
    REQUIRED_NAMED.add_argument('-ftp_folder_path', '--ftp_folder_path', action='store',
                                help='Input the FTP folder path', required=True)
    REQUIRED_NAMED.add_argument('-ftp_password', '--ftp_password', action='store',
                                help='Input the FTP password', required=True)
    REQUIRED_NAMED.add_argument('-predictions_cfg_json',
                                '--predictions_cfg_json',
                                help='Input (binary relevance) json configuration file for predictions output',
                                required=True)
    OPTIONAL_NAMED.add_argument('-download_folder_path', '--download_folder_path', action='store',
                                help='Folder path to download files',
                                default="FTP_downloaded/")
    OPTIONAL_NAMED.add_argument('-csv_filename', '--csv_filename', action='store',
                                help='Input the name of csv file to save results',
                                default=DEFAULT_CSV_FILENAME)
    OPTIONAL_NAMED.add_argument('-max_runtime_minutes', '--max_runtime_minutes', type=int,
                                help='Max run time in checking ftp in minutes',
                                default=sys.maxsize)
    OPTIONAL_NAMED.add_argument('-wait_time_minutes', '--wait_time_minutes', type=int,
                                help='Wait time in between checking ftp in minutes',
                                default=2)
    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    RESULT = ARGUMENT_PARSER.parse_args()

    MAX_MINUTES = RESULT.max_runtime_minutes
    WAIT_TIME_MINUTES = RESULT.wait_time_minutes

    while MAX_MINUTES > 0:
        start_batch_run_ftp_live(ftp_folder_path=RESULT.ftp_folder_path,
                                 ftp_password=RESULT.ftp_password,
                                 download_folder_path=RESULT.download_folder_path,
                                 predictions_cfg_json=RESULT.predictions_cfg_json,
                                 csv_filename=RESULT.csv_filename)

        print("Waiting to FTP files to get accumulate:", WAIT_TIME_MINUTES, "Minutes")
        time.sleep(WAIT_TIME_MINUTES*60)
        MAX_MINUTES -= WAIT_TIME_MINUTES
