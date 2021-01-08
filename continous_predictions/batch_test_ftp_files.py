import os
import time
import csv
import argparse
from ftplib import FTP
import pandas as pd
from . import generate_before_predict_BR


############################################################################
# Target Path or Folder in FTP
############################################################################


DESCRIPTION = "Generates the csv file with prediction results"
HELP = "Give the Required Arguments"


############################################################################
# Target Path or Folder in FTP
############################################################################
DEFAULT_CSV_FILENAME = "predictions_FTP_folder.csv"
FTP_PASSWORD = "********"


############################################################################
# Parsing argument
############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-ftp_folder_path', '--ftp_folder_path', action='store',
                    help='Input the FTP folder path')
PARSER.add_argument('-csv_filename', '--csv_filename', action='store',
                    help='Input the name of csv file to save results', default=DEFAULT_CSV_FILENAME)

RESULT = PARSER.parse_args()


############################################################################
# Parsing argument
############################################################################

if RESULT.ftp_folder_path:
    PRIMARY_PATH = RESULT.ftp_folder_path
else:
    print("Using default FTP folder Path")
    TARGET_PATH_FOLDER_FTP = "BNP/DEV5101890_2019:11:21-15:45:20/"
    PRIMARY_PATH = "/home/user-u0xzU/" + TARGET_PATH_FOLDER_FTP


############################################################################
# Check for directory locally to download
############################################################################
def check_directory_to_write_wavfiles():
    """
    creates directory to write wavfiles that are downloaded from FTP
    """
    if os.path.exists("FTP_downloaded/"):
        pass
    else:
        os.mkdir("FTP_downloaded/")


############################################################################
# List only wav files into
############################################################################
def check_for_wav_only(list_values):
    """
    Get the list of wav files .wav and .WAV format only
    """
    wav_files = []
    for each_value in list_values:
        if each_value[-3:] == "WAV"  or each_value[-3:] == "wav":
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
            time1 = ftp.voidcmd("MDTM " + name)
            count += 1
            DICT[name] = time1[4:]
    sorted_list = sorted((value, key) for (key, value) in list(DICT.items()))
    sorted_filenames = [element[0] for element in sorted_list]

    return sorted_filenames


############################################################################
# Connect ftp server
############################################################################
def call_for_ftp():
    """
    Connect to FTP and display all the wav files present in directory
    """
    global ftp
    ftp = FTP('34.211.117.196', user='user-u0xzU', passwd=FTP_PASSWORD)
    print("Connection Status : connected to FTP")
    ftp.cwd(PRIMARY_PATH)
    ex = ftp.nlst()
    wav_files_only = check_for_wav_only(ex)

    # If files are needed in the order in which they are uploaded  the uncomment the below the line
    # This will take a quite a bit of time if there are more number of files

    # wav_files_only = sorting_files_same_as_upload_order(wav_files_only)

    dataframe = pd.DataFrame()
    dataframe["FileNames"] = wav_files_only
    dataframe = dataframe.sort_values(["FileNames"], ascending=[1])
    return dataframe


############################################################################
    # Loops over the list of files in ftp & saves predictions in csv
############################################################################
def start_batch_run_ftp_live():
    """
    Writes the predicted results  on to csvfile row wise
    """
    inter_df = call_for_ftp()
    malformed = []
    dum_df = inter_df
    tag_names = ["FileNames", "Motor_probability", "Explosion_probability",
                 "Human_probability", "Nature_probability", "Domestic_probability",
                 "Tools_probability"]

    # Check if the csv file is already existing or not. If it is existing then append the result
    # to same csv file based on the downloaded file
    if os.path.exists(DEFAULT_CSV_FILENAME):
        with open(DEFAULT_CSV_FILENAME, "a") as file_object:
            wav_information_object = csv.writer(file_object)
            file_object.flush()
            for each_file in dum_df['FileNames'].tolist():
                path = "FTP_downloaded/"+each_file
                check_directory_to_write_wavfiles()
                if os.path.exists(path):
                    print("path Exists")
                else:
                    with open(path, 'wb') as file_obj:
                        ftp.retrbinary('RETR '+each_file, file_obj.write)
                    try:
                        emb = generate_before_predict_BR.main(path, 0, 0, 0)
                    except ValueError:
                        print("malformed index", dum_df.loc[dum_df["FileNames"] == each_file].index)
                        dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == each_file].index)
                        malformed.append(each_file)
                        os.remove(path)
                        continue

                    # Predict the result and save the result to the csv file
                    predictions_each_model = []
                    print("Predicting for :", each_file)
                    for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
                        pred_prob, pred = generate_before_predict_BR.main(path+each_file, 1, emb, each_model)
                        if pred_prob:
                            predictions_each_model.append("{0:.2f}".format(pred_prob[0][0] * 100))
                        else:
                            predictions_each_model.append("NaN")
                    wav_information_object.writerow([each_file] + predictions_each_model)
                    file_object.flush()

    # Is there is no csv file then create one and write the result onto it.
    else:
        with open(DEFAULT_CSV_FILENAME, "w") as file_object:
            wav_information_object = csv.writer(file_object)
            wav_information_object.writerow(tag_names)
            file_object.flush()

            # Loop over the files
            for each_file in dum_df['FileNames'].tolist():
                path = "FTP_downloaded/"+each_file
                check_directory_to_write_wavfiles()
                if os.path.exists(path):
                    print("path Exists")
                else:
                    with open(path, 'wb') as file_obj:
                        ftp.retrbinary('RETR '+ each_file, file_obj.write)
                try:
                    emb = generate_before_predict_BR.main(path, 0, 0,0)
                except ValueError:
                    print("malformed index", dum_df.loc[dum_df["FileNames"] == each_file].index)
                    dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == each_file].index)
                    malformed.append(each_file)
                    os.remove(path)
                    continue
                predictions_each_model = []
                print("Predicting for :", each_file)
                for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
                    pred_prob, pred = generate_before_predict_BR.main(path+each_file, 1, emb, each_model)
                    if pred_prob:
                        predictions_each_model.append("{0:.2f}".format(pred_prob[0][0] * 100))
                    else:
                        predictions_each_model.append("NaN")
                wav_information_object.writerow([each_file] + predictions_each_model)
                file_object.flush()


############################################################################
            # Main Function
############################################################################
if __name__=="__main__":
    while(True):
        start_batch_run_ftp_live()
        time.sleep(120)
        print("Waiting to FTP files to get accumulate: 2 Minutes")
