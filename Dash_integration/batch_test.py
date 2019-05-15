import subprocess
import os
import pickle
import base64
import datetime
import glob
import numpy as np
from ftplib import FTP
import dash
import dash.dependencies
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import generate_before_predict
import dash_table
import pymongo
import csv


# Target Path or Folder in FTP
TARGET_PATH_FOLDER_FTP = "may/"
PRIMARY_PATH = "/home/user-u0xzU/" + TARGET_PATH_FOLDER_FTP
print PRIMARY_PATH


def check_directory_to_write_wavfiles():
    """
    creates directory to write wavfiles that are downloaded from FTP
    """
    if os.path.exists("FTP_downloaded/"):
        pass
    else:
        os.mkdir("FTP_downloaded/")


def check_for_wav_only(list_values):
    """
    Get the list of wav files .wav and .WAV format only
    """
    wav_files = []
    for each_value in list_values:
        if each_value[-3:] == "WAV"  or each_value[-3:] == "wav":
            wav_files.append(each_value)
    return wav_files


def call_for_ftp():
    """
    Connect to FTP and display all the wav files present in directory
    """
    global ftp
    ftp = FTP('34.211.117.196', user='******', passwd='******')
    print "connected to FTP"
    ftp.cwd(PRIMARY_PATH)
    ex = ftp.nlst()
    wav_files_only = check_for_wav_only(ex)
    dataframe = pd.DataFrame()
    dataframe["FileNames"] = wav_files_only
    dataframe = dataframe.sort_values(["FileNames"], ascending=[1])
    return dataframe


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
    if os.path.exists("wav_file_data.csv"):
        with open("wav_file_data.csv", "a") as file_object:
            wav_information_object = csv.writer(file_object)
            file_object.flush()
            for each_file in dum_df['FileNames'].tolist():
                path = "FTP_downloaded/"+each_file
                check_directory_to_write_wavfiles()
                if os.path.exists(path):
                    print "path Exists"
                else:
                    print "Downloading and generating embeddings ", each_file
                    with open(path, 'wb') as file_obj:
                        ftp.retrbinary('RETR '+each_file, file_obj.write)
                    try:
                        emb = generate_before_predict.main(path, 0, 0)
                    except ValueError:
                        print "malformed index", dum_df.loc[dum_df["FileNames"] == each_file].index
                        dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == each_file].index)
                        malformed.append(each_file)
                        os.remove(path)

                    # Predict the result and save the result to the csv file
                    pred_prob, pred = generate_before_predict.main(path, 1, np.array(emb.flatten().tolist()))
                    motor_probability = "{0:.2f}".format(pred_prob[0][0])
                    explosion_probability = "{0:.2f}".format(pred_prob[0][1])
                    human_probability = "{0:.2f}".format(pred_prob[0][2])
                    nature_probabilty = "{0:.2f}".format(pred_prob[0][3])
                    domestic = "{0:.2f}".format(pred_prob[0][4])
                    tools = "{0:.2f}".format(pred_prob[0][5])
                    wav_information_object.writerow([each_file, motor_probability,
                                                     explosion_probability, human_probability,
                                                     nature_probabilty, domestic, tools])
                    file_object.flush()

    # Is there is no csv file then create one and write the result onto it.
    else:
        with open("wav_file_data.csv", "w") as file_object:
            wav_information_object = csv.writer(file_object)
            wav_information_object.writerow(tag_names)
            file_object.flush()

            # Loop over the files
            for each_file in dum_df['FileNames'].tolist():
                path = "FTP_downloaded/"+each_file
                check_directory_to_write_wavfiles()
                if os.path.exists(path):
                    print "path Exists"
                else:
                    print "Downloading and generating embeddings ", each_file
                    with open(path, 'wb') as file_obj:
                        ftp.retrbinary('RETR '+ each_file, file_obj.write)
                try:
                    emb = generate_before_predict.main(path, 0, 0)
                except ValueError:
                    print "malformed index", dum_df.loc[dum_df["FileNames"] == each_file].index
                    dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == each_file].index)
                    malformed.append(each_file)
                    os.remove(path)
                pred_prob, pred = generate_before_predict.main(path, 1, np.array(emb.flatten().tolist()))
                motor_probability = "{0:.2f}".format(pred_prob[0][0])
                explosion_probability = "{0:.2f}".format(pred_prob[0][1])
                human_probability = "{0:.2f}".format(pred_prob[0][2])
                nature_probabilty = "{0:.2f}".format(pred_prob[0][3])
                domestic = "{0:.2f}".format(pred_prob[0][4])
                tools = "{0:.2f}".format(pred_prob[0][5])
                wav_information_object.writerow([each_file, motor_probability,
                                                 explosion_probability, human_probability,
                                                 nature_probabilty, domestic, tools])
                file_object.flush()


if __name__=="__main__":
    start_batch_run_ftp_live()

