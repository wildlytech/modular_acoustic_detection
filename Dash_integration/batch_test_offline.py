"""
Generates the csv file with prediction results
"""
import os
import glob
import argparse
import numpy as np
import csv
import pydub
import pandas as pd
import generate_before_predict
from pydub import AudioSegment


DESCRIPTION = "Generates the csv file with prediction results"
HELP = "Give the Required Arguments"


PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-local_folder_path', '--local_folder_path', action='store',
                    help='Input the path')
PARSER.add_argument('-csv_filename', '--csv_filename', action='store',
                    help='Input the name of csv file to save results', default='wav_file_results_from_script.csv')
RESULT = PARSER.parse_args()


FOLDER_FILES_PATH = RESULT.local_folder_path
CSV_FILENAME = RESULT.csv_filename

def start_batch_run_ftp_live(path_for_folder):
    """
    Writes the predicted results  on to csvfile row wise
    """
    all_wav_files_path = glob.glob(path_for_folder+"*.WAV") + glob.glob(path_for_folder+"*.wav")
    all_wav_files = [each_file.split("/")[-1] for each_file in all_wav_files_path]
    print 'len:', len(all_wav_files)
    dum_df = pd.DataFrame()
    dum_df["FileNames"] = all_wav_files
    malformed_specific = []
    tag_names = ["FileNames", "Motor_probability", "Explosion_probability",
                 "Human_probability", "Nature_probability", "Domestic_probability",
                 "Tools_probability", "dBFS"]


    # Check if the csv file is already existing or not. If it is existing then append the result
    # to same csv file based on the downloaded file
    if os.path.exists(CSV_FILENAME):
        data_read = pd.read_csv(CSV_FILENAME)
        list_of_files_predicted = data_read['FileNames'].tolist()
        with open(CSV_FILENAME, "a") as file_object:
            wav_information_object = csv.writer(file_object)
            file_object.flush()
            for each_file in dum_df['FileNames'].tolist():
                if each_file not in list_of_files_predicted:
                    try:
                        emb = generate_before_predict.main(path_for_folder+each_file, 0, 0)
                        dbfs = AudioSegment.from_wav(path_for_folder+each_file).dBFS
                    except ValueError:
                        print "malformed index", dum_df.loc[dum_df["FileNames"] == each_file].index
                        dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == each_file].index)
                        malformed_specific.append(each_file)
                        continue

                    # Predict the result and save the result to the csv file
                    pred_prob, pred = generate_before_predict.main(path_for_folder+each_file, 1, np.array(emb.flatten().tolist()))
                    motor_probability = "{0:.2f}".format(pred_prob[0][0])
                    explosion_probability = "{0:.2f}".format(pred_prob[0][1])
                    human_probability = "{0:.2f}".format(pred_prob[0][2])
                    nature_probabilty = "{0:.2f}".format(pred_prob[0][3])
                    domestic = "{0:.2f}".format(pred_prob[0][4])
                    tools = "{0:.2f}".format(pred_prob[0][5])
                    wav_information_object.writerow([each_file, motor_probability,
                                                     explosion_probability, human_probability,
                                                     nature_probabilty, domestic, tools, dbfs])
                    file_object.flush()
                else:
                    pass

    # Is there is no csv file then create one and write the result onto it.
    else:
        with open(CSV_FILENAME, "w") as file_object:
            wav_information_object = csv.writer(file_object)
            wav_information_object.writerow(tag_names)
            file_object.flush()

            # Loop over the files
            for each_file in dum_df['FileNames'].tolist():
                try:
                    emb = generate_before_predict.main(path_for_folder+each_file, 0, 0)
                    dbfs = AudioSegment.from_wav(path_for_folder+each_file).dBFS
                except ValueError:
                    print "malformed index", dum_df.loc[dum_df["FileNames"] == each_file].index
                    dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == each_file].index)
                    malformed_specific.append(each_file)
                    continue

                pred_prob, pred = generate_before_predict.main(path_for_folder+each_file, 1, np.array(emb.flatten().tolist()))
                motor_probability = "{0:.2f}".format(pred_prob[0][0])
                explosion_probability = "{0:.2f}".format(pred_prob[0][1])
                human_probability = "{0:.2f}".format(pred_prob[0][2])
                nature_probabilty = "{0:.2f}".format(pred_prob[0][3])
                domestic = "{0:.2f}".format(pred_prob[0][4])
                tools = "{0:.2f}".format(pred_prob[0][5])
                wav_information_object.writerow([each_file, motor_probability,
                                                 explosion_probability, human_probability,
                                                 nature_probabilty, domestic, tools, dbfs])
                file_object.flush()

if __name__ == "__main__":
    start_batch_run_ftp_live(FOLDER_FILES_PATH)
