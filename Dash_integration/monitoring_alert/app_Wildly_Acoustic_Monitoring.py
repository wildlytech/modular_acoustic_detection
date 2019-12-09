"""
"""
from time import strptime
import itertools
import threading
from io import BytesIO
from ssl import SSLSocket
import struct
import time
import sys
import base64
import os
from ftplib import FTP
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from datetime import datetime
from datetime import timedelta
import urllib
import dash_table
import generate_before_predict_BR
import numpy as np
import csv
import requests
from scipy.io import wavfile




####################################################################################
                    # running directly with Python
####################################################################################
if __name__ == '__main__':
    from utils.app_standalone import run_standalone_app


####################################################################################
                # FTP path and defining soi
####################################################################################
ftp_path = "BNP/"
LABELS_ALIGNED = ["Motor Sounds", "Explosion Sounds", "Human Sounds", "Nature Sounds", "Domestic Sounds", "Tools Sounds"]
SOI = ["Motor Sounds", "Explosion Sounds", "Human Sounds", "Domestic Sounds", "Tools Sounds"]




####################################################################################
                # Predefining variables
####################################################################################
GLOBAL_STOP_THREAD = False
if os.path.exists("downloaded_audio_files/soi_csv_file.csv"):
    os.remove("downloaded_audio_files/soi_csv_file.csv")
else:
    pass



####################################################################################
         # FTP credentials & Message Authorisation token from fast2sms.com
####################################################################################
FTP_USERNAME = 'user-u0xzU'
FTP_PASSWORD = '**********'
FTP_HOST = '34.211.117.196'
AUTHORIZATION_TOKEN = "***********"


####################################################################################
                # Creates dash table format from pandas dataframe format
####################################################################################
def Table(dataframe, column_name):
    """
    Creates dash table format from pandas dataframe format
    """
    rows = []
    for i in range(len(dataframe)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i][col]
            if os.path.exists("data_downloaded/"+dataframe.iloc[i][1]+".csv"):
                csv_string = pd.read_csv("data_downloaded/"+dataframe.iloc[i][1]+".csv")
                csv_string = csv_string.to_csv(index=False,
                                               encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.quote(csv_string)
                if col == column_name:
                    cell = html.Td(html.A(id="download-report",
                                          href=csv_string,
                                          download="data_downloaded/"+dataframe.iloc[i][1]+".csv",
                                          children=[value],
                                          style={"color":"blue",
                                                 'text-decoration':'underline'}),
                                   style={"padding-top":"10px",
                                          "padding-right":"13px",
                                          "padding-left":"10px",
                                          'text-align':'center',})
                else:
                    cell = html.Td(children=value,
                                   style={"padding-top":"10px",
                                          'color':'white',
                                          "padding-right":"13px",
                                          "padding-left":"10px",
                                          'text-align':'center'})
                row.append(cell)
            else:
                if col == column_name:
                    cell = html.Td(html.A(children=[value],
                                          style={"color":"white",
                                                 'text-decoration':'underline'}),
                                   style={"padding-top":"10px",
                                          "padding-right":"13px",
                                          "padding-left":"10px",
                                          'text-align':'center'})
                else:
                    cell = html.Td(children=value,
                                   style={"padding-top":"10px",
                                          'color':'white',
                                          "padding-right":"13px",
                                          "padding-left":"10px",
                                          'text-align':'center'})
                row.append(cell)
        rows.append(html.Tr(row))

    return html.Table(
        [html.Tr([html.Th(col,
                          style={"padding-top":"10px",
                                 "padding-right":"30px",
                                 'color':'white',
                                 "padding-left":"30px",
                                 "padding-bottom":'10px',
                                 'text-align':'center'}) for col in dataframe.columns])] + rows)


####################################################################################
                # Lists all the directories with timestamps
####################################################################################
def get_directories_listed(ftp_path):
    """
    Lists all the directories with timestamps
    """
    dir_n_timestamp = []
    ftp.sendcmd("TYPE A")
    lines = []
    now = datetime.now()
    now_year = now.strftime("%Y")
    datetimeformat_1 = '%Y/%m/%d-%H:%M'
    ftp.dir(lines.append)

    for line in lines:
        if line[0] == 'd':
            directory = line.split(' ')[-1]
            if len(line.split(" ")[-2].split(":")) == 2:
                if line.split(' ')[-4]:
                    month = line.split(' ')[-4]
                else:
                    month = line.split(' ')[-5]
                timestamp1 = now_year +'/'+str(strptime(month, '%b').tm_mon)+'/'+\
                line.split(' ')[-3]+'-'+line.split(' ')[-2]
                time2 = str(datetime.strptime(timestamp1, datetimeformat_1) + timedelta(minutes=330))
                dir_n_time = directory, time2, 'active'
                dir_n_timestamp.append(dir_n_time)

            else:
                timestamp1 = line.split(' ')[-2]+'/'+str(strptime(line.split(' ')[-5], '%b').tm_mon)\
                +'/'+line.split(' ')[-4]
                dir_n_time = directory, timestamp1, 'inactive'
                dir_n_timestamp.append(dir_n_time)

    return dir_n_timestamp, [each_[0] for each_ in dir_n_timestamp], [each_[1] for each_ in dir_n_timestamp]


####################################################################################
            # Returns ftp timestamps
####################################################################################
def last_ftp_time(ftp_path):
    """
    """
    datetimeformat_2 = '%Y-%m-%d %H:%M:%S'
    now = datetime.now()
    directories_time_list_inside_scope = []
    timestamp2 = now.strftime(datetimeformat_2)
    dir_n_timestamp_inside_scope, _, _ = get_directories_listed(ftp_path)
    for dir_n_time_inside_scope in dir_n_timestamp_inside_scope:
        if dir_n_time_inside_scope[2] == 'active':
            if len(dir_n_time_inside_scope[1].split(' ')) != 1:
                time_diff = datetime.strptime(timestamp2, datetimeformat_2) - datetime.strptime(dir_n_time_inside_scope[1], datetimeformat_2)
                directories_time_list_inside_scope.append(str(time_diff))
        else:
            directories_time_list_inside_scope.append('inactive')
    return dir_n_timestamp_inside_scope, directories_time_list_inside_scope


####################################################################################
            # Returns directories / device active or inactive
####################################################################################
def active_or_inactive(dir_n_timestamp, directories_time_list):
    """
    Returns directories / device active or inactive
    """
    status = []
    for td in directories_time_list:
        if td == 'inactive':
            status.append('Inactive')
        else:
            if len(td.split(' ')) == 1:
                hours = int(td.split(":")[0])
                month = int(td.split(":")[1])
                seconds_ = int(td.split(":")[2])
                seconds = hours*60*60 + month*60 + seconds_
            else:
                days = td.split(" ")[0]
                hours = td.split(" ")[2].split(":")[0]
                month = td.split(" ")[2].split(":")[1]
                seconds_ = td.split(" ")[2].split(":")[2]
                seconds = days*24*60*60 + hours*60*60 + month*60 + seconds_
            if seconds <= 300:
                status.append('Active')
            else:
                status.append('Inactive')
    return dir_n_timestamp, status



####################################################################################
            # Connect to FTP server
####################################################################################
def connect(ftp_path):
    '''
    To connect to ftp
    '''
    global ftp, done
    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
    done = True
    ftp.cwd(ftp_path)



####################################################################################
        # Connecting ftp server and listing all directories
####################################################################################
connect(ftp_path)
DIR_AND_TIME, DIRECTORIES, TIMESTAMPS = get_directories_listed(ftp_path)
DIR_AND_TIMESTAMP, DIRECTORIES_TIME_LIST = last_ftp_time(ftp_path)
DIR_AND_TIMESTAMP, STATUS = active_or_inactive(DIR_AND_TIMESTAMP, DIRECTORIES_TIME_LIST)
DATAFRAME_REQUIRED = pd.DataFrame()



####################################################################################
            # Returns the deviceid from directory name
####################################################################################
def get_devid_from_dir(dir_name):
    device_id = dir_name.split("_")[0][3:]
    return device_id


####################################################################################
            # Creatin a dataframe to display
####################################################################################
DEVIDS = []
for each in DIRECTORIES:
    DEVIDS.append(get_devid_from_dir(each))
DATAFRAME_REQUIRED['Directories'] = DIRECTORIES
DATAFRAME_REQUIRED["TimeStamps"] = TIMESTAMPS
DATAFRAME_REQUIRED["Status"] = STATUS



####################################################################################
            # Colors to headers
####################################################################################
def header_colors():
    """
    Colors to headers
    """
    return {
        'bg_color': '#232323',
        'font_color': 'white'
    }


####################################################################################
                # Loop through all the models and get predictions
####################################################################################
def predictions_from_models(wavfile_path, embeddings):
    """
    """
    prediction_list = []
    prediction_rounded = []
    for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
        pred_prob, pred_round = generate_before_predict_BR.main(wavfile_path, 1, embeddings, each_model)
        pred_prob = pred_prob[0][0] * 100
        pred_round = pred_round[0][0]
        prediction_list.append("{0:.2f}".format(pred_prob))
        prediction_rounded.append(pred_round)
    return prediction_list, prediction_rounded



####################################################################################
        # Generates embeddings for each file and calls for predictions
####################################################################################
def get_predictions(wavfile_path):
    """
    """
    try:
        embeddings = generate_before_predict_BR.main(wavfile_path, 0, 0, 0)
    except:
        print('\033[1m'+ "Predictions: " + '\033[0m' + "Error occured in File:- " + wavfile_path.split("/")[-1])
        return None, None
    try:
        prediction_list = []
        prediction_rounded = []
        for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
            pred_prob, pred_round = generate_before_predict_BR.main(wavfile_path, 1, embeddings, each_model)
            pred_prob = pred_prob[0][0] * 100
            pred_round = pred_round[0][0]
            prediction_list.append("{0:.2f}".format(pred_prob))
            prediction_rounded.append(pred_round)

        return prediction_list, prediction_rounded
    except OSError:
        return None, None


####################################################################################
                    # Saves the audio file / Downloads audio file
####################################################################################
def save_file(name, content):
    """
    Decode and store a file uploaded with Plotly Dash.
    """
    data = content.encode("utf8").split(b";base64,")[1]
    with open("FTP_downloaded/"+name, "wb") as file_p:
        file_p.write(base64.b64decode(data))


####################################################################################
             # Removes the duplicate from looping
####################################################################################
def get_without_duplicates(dir_name, csv_filename, ftp_obj):
    """
    """
    sorted_wavfiles = ftp_obj.nlst("*.wav")
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)["Filename"].values.tolist()
        if df:
            non_repeated = []
            for each_value in sorted_wavfiles:
                if each_value not in df:
                    non_repeated.append(each_value)
                else:
                    pass
            return non_repeated
        return sorted_wavfiles
    else:
        return sorted_wavfiles


####################################################################################
            # Creates a directory if not already present
####################################################################################
def check_path_for_downloading(wavfile_path):
    """
    """
    if os.path.exists(wavfile_path):
        return
    else:
        os.mkdir(wavfile_path)
        return


####################################################################################
            # sorts ftp files based on names
####################################################################################
def sort_on_filenames(files_list):
    """
    """
    only_wavfiles = []
    wav_files_list = []
    wav_files_number = []
    for name in files_list:
        if (name[-3:] == 'wav') or (name[-3:] == 'WAV'):
            character = name[0:1]
            extension = name[-4:]
            only_wavfiles.append(name)
            wav_files_number.append("".join(name[1:-4].split("_")))
        else:
            pass
    if wav_files_number:
        wav_files_toint = map(int, wav_files_number)
        sorted_wavfiles = sorted(wav_files_toint)

        for sorted_file in sorted_wavfiles:
            wav_files_list.append(character + str(sorted_file)[:8]+"_"+ str(sorted_file)[8:]+ extension)
        return wav_files_list
    else:
        return None



####################################################################################
        # Downloads audio files but checks for already existed one
####################################################################################
def download_files(each_wav_file, path_to_download, blockalign, samplerate,ftp_obj):
    '''
    downloads single wav file
    '''
    if os.path.exists(path_to_download+each_wav_file):
        if os.path.getsize(path_to_download+each_wav_file) > samplerate*blockalign*10:
            pass
        else:
            with open(path_to_download+each_wav_file, 'wb') as file_obj:
                ftp_obj.retrbinary('RETR '+ each_wav_file, file_obj.write)

    else:
        with open(path_to_download+each_wav_file, 'wb') as file_obj:
            ftp_obj.retrbinary('RETR '+ each_wav_file, file_obj.write)



####################################################################################
            # Writes csv with all the information
####################################################################################
def write_csv_prediction(direct_name, filename, device_id, prediction_list, csv_filename):
    """
    """
    if os.path.exists(csv_filename):
        with open(csv_filename, "a") as file_object:
            wav_information_object = csv.writer(file_object)
            wav_information_object.writerow([direct_name,
                                             filename,
                                             device_id,
                                             prediction_list[0],
                                             prediction_list[1],
                                             prediction_list[2],
                                             prediction_list[3],
                                             prediction_list[4],
                                             prediction_list[5]])
            file_object.flush()

    else:
        header_names = ["Directory",
                        "Filename",
                        "DeviceID",
                        "Motor",
                        "Explosion",
                        "Human",
                        "Nature",
                        "Domestic",
                        "Tools"]
        with open(csv_filename, "w") as file_object:
            wav_information_object = csv.writer(file_object)
            wav_information_object.writerow(header_names)
            file_object.flush()
            wav_information_object.writerow([direct_name,
                                             filename,
                                             device_id,
                                             prediction_list[0],
                                             prediction_list[1],
                                             prediction_list[2],
                                             prediction_list[3],
                                             prediction_list[4],
                                             prediction_list[5]])
            file_object.flush()



####################################################################################
        # Checks if the predicted sounds consist of sound iof interest
####################################################################################
def check_for_soi(prediction_list_rounded, LABELS_ALIGNED):
    """
    """
    global SOI
    soi_predicted = []
    for index, each_pred in enumerate(prediction_list_rounded):
        if each_pred:
            pred_label_name = LABELS_ALIGNED[index]
            if pred_label_name in SOI:
                soi_predicted.append(pred_label_name)
            else:
                pass
        else:
            pass
    return soi_predicted





####################################################################################
    # Loops over all the directories selected for going through process
####################################################################################
def directory_procedure(directory_listing, selected_device, ftp_obj):
    """
    """
    global done
    ftp_obj.cwd(ftp_path)
    print(ftp_obj.pwd())
    ftp_obj.cwd(directory_listing[0]+"/")
    for each_ in directory_listing:
        while True:
            wavfiles = get_without_duplicates(each_,
                                              "downloaded_audio_files/"+each_+".csv",
                                              ftp_obj)
            wavfiles = sort_on_filenames(wavfiles)
            if wavfiles:
                for each_file in wavfiles:
                    wavheader, _ = get_wavheader_extraheader(each_file, ftp_obj)
                    if check_ftp_wav_file_size(each_file,
                                               wavheader["BlockAlign"],
                                               wavheader["SampleRate"],
                                               ftp_obj):
                        check_path_for_downloading("downloaded_audio_files/"+each_)
                        download_files(each_file,
                                       "downloaded_audio_files/"+each_+"/",
                                       wavheader["BlockAlign"],
                                       wavheader["SampleRate"],
                                       ftp_obj)
                        prediction_list, prediction_rounded = get_predictions("downloaded_audio_files/"+each_+"/"+each_file)
                        if prediction_list:
                            write_csv_prediction(each_,
                                                 each_file,
                                                 selected_device,
                                                 prediction_list,
                                                 "downloaded_audio_files/"+each_+".csv")
                            decide_to_alert = check_for_soi(prediction_rounded, LABELS_ALIGNED)
                            print "decide to alert:",decide_to_alert
                            if decide_to_alert:
                                write_csv_prediction(each_,
                                                     each_file,
                                                     selected_device,
                                                     prediction_list,
                                                     "downloaded_audio_files/"+"soi_csv_file.csv")
                                get_alerted("downloaded_audio_files/"+each_+"/"+each_file,
                                            decide_to_alert,
                                            prediction_list,
                                            None)
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
            else:
                pass


######################################################################################
                    # Loops over FTP directories for processing: Threads
######################################################################################
def directory_procedure_threading(directory_listing, selected_device, ftp_obj, phoneno):
    """
    Loops over directories for processing
    """
    global GLOBAL_STOP_THREAD
    global done
    ftp_obj.cwd(ftp_path)
    print(ftp_obj.pwd())
    ftp_obj.cwd(directory_listing[0]+"/")
    for each in directory_listing:
        while not GLOBAL_STOP_THREAD:
            wavfiles = get_without_duplicates(each,
                                              "downloaded_audio_files/"+each+".csv",
                                              ftp_obj)
            wavfiles = sort_on_filenames(wavfiles)
            if wavfiles:
                for each_file in wavfiles:
                    if not GLOBAL_STOP_THREAD:
                        wavheader, _ = get_wavheader_extraheader(each_file, ftp_obj)
                        if check_ftp_wav_file_size(each_file,
                                                   wavheader["BlockAlign"],
                                                   wavheader["SampleRate"],
                                                   ftp_obj):
                            check_path_for_downloading("downloaded_audio_files/"+each)
                            download_files(each_file,
                                           "downloaded_audio_files/"+each+"/",
                                           wavheader["BlockAlign"],
                                           wavheader["SampleRate"],
                                           ftp_obj)
                            prediction_list, prediction_rounded = get_predictions("downloaded_audio_files/"+each+"/"+each_file)
                            if prediction_list:
                                write_csv_prediction(each,
                                                     each_file,
                                                     selected_device,
                                                     prediction_list,
                                                     "downloaded_audio_files/"+each+".csv")
                                decide_to_alert = check_for_soi(prediction_rounded, LABELS_ALIGNED)
                                print "decide to alert:", decide_to_alert
                                if decide_to_alert:
                                    write_csv_prediction(each,
                                                         each_file,
                                                         selected_device,
                                                         prediction_list,
                                                         "downloaded_audio_files/"+"soi_csv_file.csv")
                                    get_alerted("downloaded_audio_files/"+each+"/"+each_file,
                                                decide_to_alert,
                                                prediction_list,
                                                phoneno)
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass
                    else:
                        break
            else:
                pass




####################################################################################
                # Alert function
####################################################################################
def get_alerted(wavfile_path, soi_found, prediction_list, phoneNo):
    """
    Message Text
    WavFile Name:
    Predicted Sounds:
    Device Location:
    TimeStamp-File:
    """
    if phoneNo:
        get_text = ""
        for each_ in soi_found:
            if "Motor" in each_.split(" "):
                get_text = get_text +"Motor,"
            if "Explosion" in each_.split(" "):
                get_text = get_text+ "Explosion,"
            if "Human" in each_.split(" "):
                get_text = get_text + "Human,"
            if "Tools" in each_.split(" "):
                get_text = get_text+ "Tools,"


        wavfile_ = wavfile_path.split("/")[-1]

        text = "\nWavFile-"+wavfile_ + "\n, Sound(s) Found- "+get_text
        url = "https://www.fast2sms.com/dev/bulk"
        payload = "sender_id=FSTSMS&message="+ text+"&language=english&route=p&numbers="+phoneNo
        headers = {'authorization': AUTHORIZATION_TOKEN,
                   'Content-Type': "application/x-www-form-urlencoded",
                   'Cache-Control': "no-cache"}
        response = requests.request("POST", url, data=payload, headers=headers)
        print(response.text)
        return text
    else:
        return



####################################################################################
                # Checks for FTP size
####################################################################################
def check_ftp_wav_file_size(each_wav_file, blockalign, samplerate,ftp_obj):
    '''
    checks wav file size
    '''
    ftp_obj.sendcmd("TYPE i")
    if ftp_obj.size(each_wav_file) > samplerate*blockalign*10:
        ftp_obj.sendcmd("TYPE A")
        return True
    else:
        return False


class FtpFile:
    """
    sdfsg
    """

    def __init__(self, ftp_, name):
        """
        """
        self.ftp_ = ftp_
        self.name = name
        self.size = 10240
        self.pos = 0

    def seek(self, offset, whence):
        """
        """
        if whence == 0:
            self.pos = offset
        if whence == 1:
            self.pos += offset
        if whence == 2:
            self.pos = self.size + offset

    def tell(self):
        """
        """
        return self.pos

    def read(self, size= None):
        """
        """
        if size == None:
            size = self.size - self.pos
        data = B""
        self.ftp_.voidcmd('TYPE I')
        cmd = "RETR {}".format(self.name)
        conn = self.ftp_.transfercmd(cmd, self.pos)
        try:
            while len(data) < size:
                buf = conn.recv(min(size - len(data), 8192))
                if not buf:
                    break
                data += buf
            if SSLSocket is not None and isinstance(conn, SSLSocket):
                conn.unwrap()
        finally:
            conn.close()
        try:
            self.ftp_.voidresp()
        except:
            pass
        self.pos += len(data)
        return data




####################################################################################
            # returns wavheader and extraheader information
####################################################################################
def get_wavheader_extraheader(name,ftp_obj):
    '''
    To read wav file header details
    '''
    wavheader_dict = {}
    if (name[-3:] == 'wav') or (name[-3:] == 'WAV'):

        file_header_info = BytesIO(FtpFile(ftp_obj, name).read(264))

        riff, size, fformat = struct.unpack('<4sI4s', file_header_info.read(12))
        chunkoffset = file_header_info.tell()

        chunk_header = file_header_info.read(8)
        subchunkid, subchunksize = struct.unpack('<4sI', chunk_header)
        chunkoffset = file_header_info.tell()

        aformat, channels, samplerate, byterate, blockalign, bps = struct.unpack('HHIIHH', \
            file_header_info.read(16))
        chunkoffset = file_header_info.tell()

        struct.unpack('<4sI', file_header_info.read(8))
        struct.unpack('<4s4sI', file_header_info.read(12))
        chunkoffset = file_header_info.tell()

        extra_header = file_header_info.read(200)
        chunkoffset = file_header_info.tell()

        file_header_info.seek(chunkoffset)
        subchunk2id, subchunk2size = struct.unpack('<4sI', file_header_info.read(8))
        chunkoffset = file_header_info.tell()

        wav_header = [riff, size, fformat, subchunkid, subchunksize, aformat, \
        channels, samplerate, byterate, blockalign, bps, subchunk2id, subchunk2size]

        for each_value in zip(wav_header, ["ChunkID", "TotalSize", "Format", "SubChunk1ID", \
            "SubChunk1Size", "AudioFormat", "NumChannels", "SampleRate", "ByteRate", \
            "BlockAlign", "BitsPerSample", "SubChunk2ID", "SubChunk2Size"]):
            wavheader_dict[each_value[1]] = each_value[0]


        extra_header_info = extra_header.decode("ascii").split(',')
        return wavheader_dict, extra_header_info



####################################################################################
                # Parse the single file input
####################################################################################
def parse_contents(contents, filename, date):
    """
    Read the file contents
    """
    if filename[-3:] == 'wav' or 'WAV':
        if not os.path.exists("FTP_downloaded/"):
            os.makedirs("FTP_downloaded/")
        save_file(filename, contents)
        encoded_image_uploaded_file = 'FTP_downloaded/'+ filename
        encoded_image_uploaded_file = base64.b64encode(open(encoded_image_uploaded_file, 'rb').read())
        prediction_list, _ = get_predictions('FTP_downloaded/'+ filename)
        print "Predictions:", prediction_list
        if prediction_list:
            return plot_prediction_graph('FTP_downloaded/'+ filename)
        else:
            return html.Div(html.P("Error occured. Please try Again or Input proper Wav File.",
                                   style={"textAlign":"center"}))



####################################################################################
        # Prediction plot
####################################################################################
def plot_prediction_graph(filepath):
    """
    Read the file contents
    """
    encoded_image_uploaded_file = filepath
    encoded_image_uploaded_file = base64.b64encode(open(encoded_image_uploaded_file, 'rb').read())
    prediction_list, prediction_rounded = get_predictions(filepath)

    if prediction_list:
        return  html.Div(style={'color': 'green', 'fontSize':14}, children=[
            html.Div(children=[
            html.Audio(id='myaudio',
                       src='data:audio/WAV;base64,{}'.format(encoded_image_uploaded_file),
                       controls=True,
                       )],style={"margin-left":"27%"}),
            dcc.Graph(id='example',
                      figure={
                          'data':[{
                              'x':["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"],
                              'y':[i for i in prediction_list],
                              "text": [i for i in prediction_list],
                              "textposition":"auto",
                              'type':'bar'}],
                          'layout': {
                              'title':'probablistic prediction graph ',
                              'titlefont':{
                                  'family':'Courier New, monospace',
                                  'size':22,
                                  'color':'#e4e4e4'},

                              'xaxis':{
                                  'title': 'Labels of the sound',
                                  'titlefont':{
                                      'family':'Courier New, monospace',
                                      'size':18,
                                      'color': '#e4e4e4'}},
                              'yaxis':{
                                  'title': 'Percenatge probabality',
                                  'range':[0, 100],
                                  'dtick':20,
                                  'autorange':False,
                                  'titlefont':{
                                      'family':'Courier New, monospace',
                                      'size':18,
                                      'color':'#e4e4e4'}},
                              'paper_bgcolor':'#232323',
                              'plot_bgcolor':'#232323',
                              'font': {
                                  'color':'#e4e4e4'}}},
                      config={"staticPlot":True},
                      style={'marginBottom': 0,
                             'marginTop': 10}),
            html.Div(children=[
            html.P('Uploaded File : '+ filepath.split("/")[-1],
                   style={'color': '#FFBF01',
                          'fontSize': 12,
                          "text-align":"center"})])])
    else:
        return html.Div(html.P("Error occured. Please try Again or Input proper Wav File.",
                               style={"textAlign":"center"}))



####################################################################################
                # Batch of FTP files input
####################################################################################
def table_for_ftp_batch_files(dataframe, list_of_malformed):
    """
    Returns the predicted values as the data table
    """
    if list_of_malformed:
        list_of_malformed = str(list_of_malformed)
    else:
        list_of_malformed = "None"
    return html.Div([
        html.Br(),
        html.P("Total Number of Audio Clips : "+ str(dataframe.shape[0]),
               style={"color":"white",
                      'text-decoration':'underline'}),
        html.P("Error while prediciton: " + list_of_malformed,
               style={"color":"white"}),
        html.Hr(),
        dash_table.DataTable(id='datatable-interactivity-predictions',
                             columns=[{"name": i,
                                       "id": i,
                                       "deletable": True} for i in dataframe.columns],
                             data=dataframe.to_dict("rows"),
                             style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                           "fontWeight": "bold",
                                           'border': '1px solid white'},
                             style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                         'color': 'white'},
                             row_selectable="single",
                             selected_rows=[]),
        html.Hr(style={"marginTop":"30px"}),
        html.Br()])



####################################################################################
            # Color coding for selected labels
####################################################################################
def get_colored_for_soi_columns(value_list):
    """
    """
    list_output = []
    if value_list:
        for each_ in value_list:
            dictionary = {}
            dictionary['if'] = {"column_id":each_}
            dictionary['backgroundColor'] = "#3D9970"
            dictionary['color'] = 'black'
            list_output.append(dictionary)
        return list_output
    else:
        return None




####################################################################################

####################################################################################
def call_for_data_offline(dataframe, list_of_malformed):
    """
    Returns the predicted values as the data table
    """
    if list_of_malformed:
        list_of_malformed = str(list_of_malformed)
    else:
        list_of_malformed = "None Found"
    return html.Div([
        html.P("Total Number of Audio Clips : "+ str(dataframe.shape[0]),
               style={"color":"white",
                      'text-decoration':'underline'}),
        html.P("Error while prediciton: " + list_of_malformed, style={"color":"white"}),
        html.Hr(),
        dash_table.DataTable(id='datatable-interactivity-predictions',
                             columns=[{"name": i,
                                       "id": i,
                                       "deletable": True} for i in dataframe.columns],
                             data=dataframe.to_dict("rows"),
                             style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                           "fontWeight": "bold",
                                           'border': '1px solid white'},
                             style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                         'color': 'white'},
                             row_selectable="single",
                             style_table={"maxHeight":"350px",
                                          "overflowY":"scroll"},
                             selected_rows=[]),
        html.Hr()])




####################################################################################
                # Parsing batch of files uploaded
####################################################################################
def parse_contents_batch(contents, names, dates):
    """
    Multiple files that are uploaded are handled
    """
    emb = []
    malformed = []
    dum_df = pd.DataFrame()
    dum_df['FileNames'] = names
    for i in zip(contents, names, dates):
        path = "FTP_downloaded/"+i[1]
        if os.path.exists(path):
            print "path Exists"
        else:
            print "Downloading and generating embeddings ", i[1]
            save_file(i[1], i[0])
        try:
            emb.append(generate_before_predict_BR.main(path, 0, 0, 0))
        except ValueError:
            print "malformed index", dum_df.loc[dum_df["FileNames"] == i[1]].index
            dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == i[1]].index)
            malformed.append(i[1])
    dum_df['features'] = emb
    if len(dum_df["FileNames"].tolist()) == 1:
        try:
            prediction_list = []
            prediction_rounded = []
            pred_df = pd.DataFrame(columns=["File Name",
                                            "Motor",
                                            "Explosion",
                                            "Human",
                                            "Nature",
                                            "Domestic",
                                            "Tools"])
            for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
                pred_prob, pred_round = generate_before_predict_BR.main("FTP_downloaded/"+dum_df["FileNames"].tolist()[0], 1, dum_df["features"].tolist()[0], each_model)
                pred_prob = pred_prob[0][0] * 100
                pred_round = pred_round[0][0]
                prediction_list.append("{0:.2f}".format(pred_prob))
                prediction_rounded.append(pred_round)
            pred_df.loc[0] = [dum_df["FileNames"].tolist()[0]]+ prediction_list
            return call_for_data_offline(pred_df, malformed)
        except:
            return html.Div(html.P("Got Error while Predictions"))
    else:
        pred_df = pd.DataFrame(columns=["File Name", "Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"])
        for index, each_file, each_embeddings in zip(range(0, dum_df.shape[0]), dum_df["FileNames"].tolist(), dum_df["features"].values.tolist()):
            try:
                prediction_list = []
                prediction_rounded = []
                for each_model in ["Motor", "Explosion", "Human", "Nature", "Domestic", "Tools"]:
                    pred_prob, pred_round = generate_before_predict_BR.main("FTP_downloaded/"+each_file, 1, each_embeddings, each_model)
                    pred_prob = pred_prob[0][0] * 100
                    pred_round = pred_round[0][0]
                    print(pred_prob)
                    prediction_list.append("{0:.2f}".format(pred_prob))
                    prediction_rounded.append(pred_round)

                pred_df.loc[index] = [each_file] + prediction_list
            except:
                pass

        return call_for_data_offline(pred_df, malformed)




####################################################################################
                # Defining the layout
####################################################################################
def layout():

    return html.Div(id='clustergram-body', className='app-body',
        children=[
            html.Div(id='clustergram-control-tabs', className='control-tabs', 
            children=[
            dcc.Tabs(id='clustergram-tabs', value='graph', 
                children=[
                ####################################################################################
                                # Monitoring Tab
                ####################################################################################
                dcc.Tab(
                    label='Monitor Device',
                    value='graph',
                    children=[
                        html.Div(className='control-tab', children=[
                            html.Div('Select DeviceID',
                                     title="Available directories from "+ ftp_path + "  directory",
                                     className='fullwidth-app-controls-name',
                                     style={"fontWeight":"bold",
                                            "color": '#FFBF01',
                                            "text-decoration":"underline"}),
                            html.Div(id='display-all-directories-div-graph-tab',
                                     children=[dash_table.DataTable(id='display-all-directories-datatable-graph-tab',
                                                                    columns=[{"name": i,
                                                                              "id": i,} for i in DATAFRAME_REQUIRED.columns],
                                                                    data=DATAFRAME_REQUIRED.to_dict("rows"),
                                                                    row_selectable="multi",
                                                                    style_table={"Height":"500px",
                                                                                 "Width" :"500px"},
                                                                    style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                                                  "fontWeight": "bold",
                                                                                  'border': '1px solid #e4e4e4'},
                                                                    style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                                                'color': 'white'},
                                                                    style_data_conditional=[{'if': {'column_id': 'Status',
                                                                                                    'filter_query': '{Status} eq "Active"'},
                                                                                             'backgroundColor': '#3D9970',
                                                                                             'color': 'black'}])],
                                     style={"marginTop":"20px"}),
                            html.Br(),
                            html.Div(className='app-controls-block', children=[
                                html.Div('Select Alert Based on Labels:',
                                         className='app-controls-name',
                                         style={"fontSize":"10px",
                                                "width":"155px",
                                                "color":"white",
                                                "fontWeight":"bold"}),
                                dcc.Dropdown(id='sound-labels-dropdown-alert-tab',
                                             options=[{'label': 'Motor', 'value': 'Motor'},
                                                      {'label': 'Explosion', 'value': 'Explosion'},
                                                      {'label': 'Human', 'value': 'Human'},
                                                      {'label': 'Nature', 'value': 'Nature'},
                                                      {'label': 'Domestic', 'value': 'Domestic'},
                                                      {'label': 'Tools', 'value': 'Tools'}],
                                             multi=True,
                                             value=None),
                                html.Br(),
                                html.Div('Select Alert Interval :',
                                         className='app-controls-name',
                                         style={"fontSize":"10px",
                                                "color":"white",
                                                "fontWeight":"bold"}),
                                dcc.Dropdown(id='time-interval-labels-dropdown-alert-tab',
                                             options=[{'label': 'Every Event', 'value': 'all_event'},
                                                      {'label': 'Every 5Mins', 'value': '5_mins'}],
                                             multi=False,
                                             value=None),
                                html.Br(),
                                html.Br(),
                                html.Div('Alert SMS To [ +91- ]:',
                                         className='app-controls-name',
                                         style={"fontSize":"10px",
                                                "color":"white",
                                                "fontWeight":"bold"}),
                                dcc.Input(id='phone-number-alert-tab')]),
                            html.Hr(style={"marginTop":"20px"}),
                            html.Div(className='app-controls-block', children=[
                                html.Button(
                                    id='selected-all-active-devices-graph-tab',
                                    children='Monitor Active Devices',
                                    n_clicks=0,
                                    n_clicks_timestamp=0,
                                    style={"border":"1px solid #FFBF01", "color":"white"}),
                                html.Button(
                                    id='Selected-devices-button-graph-tab',
                                    children='Monitor Selected Devices',
                                    n_clicks=0,
                                    n_clicks_timestamp=0,
                                    style={"border":"1px solid #FFBF01", "color":"white"})]),
                            html.Div(id="graph-output-graph-tab-do-nothing")])]),

                ####################################################################################
                        # Upload Tab
                ####################################################################################
                dcc.Tab(
                    label='Upload Data',
                    value='datasets',
                    children=html.Div(className='control-tab', children=[
                        html.Div(id='file-upload-name'),
                        html.Div(id='clustergram-file-upload-container',
                                 title='Upload your here.',
                                 children=[
                                     dcc.Upload(id='upload-data', multiple=True, children=[
                                         html.A(html.Button(className='control-download',
                                                            children=html.Div(["Drag and Drop or click \
                                                                               to select files."],
                                                                              style={"font-size":10})))],
                                                accept='.wav')]),

                        html.Div([
                            html.A(
                                html.Button(
                                    'Download sample data',
                                    id='clustergram-download-sample-data',
                                    className='control-download'))]),
                        html.Div(id='clustergram-info')])),

                ####################################################################################
                        # FTP server Tab
                ####################################################################################
                dcc.Tab(
                    label='FTP Server',
                    value='what-is',
                    children=html.Div(className='control-tab', children=[
                        html.Div(
                            'Available Directories',
                            title="Available directories from "+ ftp_path + "  directory",
                            className='fullwidth-app-controls-name',
                            style={"fontWeight":"bold",
                                   "color": '#FFBF01',
                                   "text-decoration":"underline"}),
                        html.Div(id='display-all-directories-div',
                                 children=[dash_table.DataTable(id='display-all-directories-datatable',
                                                                columns=[{"name": i,
                                                                          "id": i,} for i in DATAFRAME_REQUIRED.columns],
                                                                data=DATAFRAME_REQUIRED.to_dict("rows"),
                                                                row_selectable="single",
                                                                style_table={"Height":"500px",
                                                                             "Width" :"500px"},
                                                                style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                                              "fontWeight": "bold",
                                                                              'border': '1px solid #e4e4e4'},
                                                                style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                                            'color': 'white'},
                                                                style_data_conditional=[{'if': {'column_id': 'Status',
                                                                                                'filter_query': '{Status} eq "Active"'},
                                                                                         'backgroundColor': '#3D9970',
                                                                                         'color': 'black'}])],
                                 style={"marginTop":"20px"}),
                        html.Br(),
                        dcc.Loading(className='dashbio-loading',
                                    type="dot",
                                    children=[html.Div(id='display-directory-on-status-selection')]),
                        html.Div(id='display-directory-on-deviceid-selection'),
                        html.Div(id='display-directory-on-timestamp-selection'),
                        html.Div(id='display-directory-on-location-selection'),
                        html.Div(id='display-selection-buttons-single-monitor')])
                )
                ])]),

            ####################################################################################
                    # Right side DIV element
            ####################################################################################
            html.Div(id='clustergram-page-content-right',
                     className='control-tabs-right',
                     children=[
                         html.Br(),
                         dcc.Loading(className='dashbio-loading',
                                     type="graph",
                                     children=[html.Div(id='graph-output-any')]),
                         html.Div(id='prediction-audio-upload'),
                         dcc.Loading(className='dashbio-loading',
                                     type="graph",
                                     children=[html.Div(id='graph-output-ftp')]),
                         html.Div(id='prediction-audio'),
                         html.Div([dcc.Loading(className='dashbio-loading',
                                               type="graph",
                                               children=[html.Div(id='graph-output-graph-tab')]),
                                   html.Br(),
                                   html.Div(id="button-stop-threads")]),
                         dcc.Loading(className='dashbio-loading',
                                     type="dot",
                                     children=[dcc.Interval(id='interval-component',
                                                            interval=10*1000, # in milliseconds
                                                            n_intervals=0)])])])

####################################################################################
        # Callback for each tab operation
####################################################################################
def callbacks(_app):
    """
    All the callbacks grouped in this function
    """

    ####################################################################################
                    # Display based on selection
    ####################################################################################
    @_app.callback(
        Output('display-all-directories-div', 'children'),
        [Input('clustergram-datasets1', 'value')])
    def get_selected_filter(value):
        """
        Display based on selection 
        """
        if value == "status":
            DATAFRAME_DEVICE_STATUS = pd.DataFrame()
            DATAFRAME_DEVICE_STATUS["Select Status"] = ["Active", "Inactive"]
            return dash_table.DataTable(id='display-all-status-active-inactive',
                                        columns=[{"name": i,
                                                  "id": i,}
                                                 for i in DATAFRAME_DEVICE_STATUS.columns],
                                        data=DATAFRAME_DEVICE_STATUS.to_dict("rows"),
                                        row_selectable="single",
                                        style_table={"maxHeight":"500px",
                                                     "maxWidth" :"400px"},
                                        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                        style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                    'color': 'white'},)
        else:
            return   dash_table.DataTable(id='display-all-directories-datatable',
                                          columns=[{"name": i,
                                                    "id": i,}
                                                   for i in DATAFRAME_REQUIRED.columns],
                                          data=DATAFRAME_REQUIRED.to_dict("rows"),
                                          row_selectable="single",
                                          style_table={"Height":"500px",
                                                       "Width" :"500px"},
                                          style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                        "fontWeight": "bold",
                                                        'border': '1px solid white'},
                                          style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                      'color': 'white'},
                                          style_data_conditional=[{'if': {'column_id': 'Status',
                                                                          'filter_query': '{Status} eq "Active"'},
                                                                   'backgroundColor': '#3D9970',
                                                                   'color': 'black'}])



    ####################################################################################
                # Play selected audio   
    ####################################################################################
    @_app.callback(
        Output('prediction-audio', 'children'),
        [Input('datatable-interactivity-predictions', 'data'),
         Input('datatable-interactivity-predictions', 'columns'),
         Input("datatable-interactivity-predictions", "derived_virtual_selected_rows")])
    def play_button_for_prediction(rows, columns, indices):
        """
        Playing the audio when file is selected
        """
        pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        if indices is not None and indices != []:
            path = "FTP_downloaded/"+str(pred_df.iloc[indices[0]]["File Name"])
            encoded_image_to_play = base64.b64encode(open(path, 'rb').read())
            return html.Div([
                html.Br(),
                html.Audio(id='myaudio',
                           src='data:audio/WAV;base64,{}'.format(encoded_image_to_play),
                           controls=True,
                           style={"margin-left":"25%"})])


    ####################################################################################
                # Disabling div element not required
    ####################################################################################
    @_app.callback(
        Output('prediction-audio', 'style'),
        [Input('clustergram-tabs', 'value')])
    def disable_ftp_graph_on_selecting_tabs_audio(value):
        """
        """
        if value == "datasets" or value == "graph":
            return {"display":"none"}


    ####################################################################################
                # Play option for predicted and selected file
    ####################################################################################
    @_app.callback(
        Output('prediction-audio-upload', 'children'),
        [Input('datatable-interactivity-predictions', 'data'),
         Input('datatable-interactivity-predictions', 'columns'),
         Input("datatable-interactivity-predictions", "derived_virtual_selected_rows")])
    def play_button_for_prediction_upload(rows, columns, indices):
        """
        Playing the audio when file is selected
        """
        pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        if indices is not None and indices != []:
            path = "FTP_downloaded/"+str(pred_df.iloc[indices[0]]["File Name"])
            encoded_image_to_play = base64.b64encode(open(path, 'rb').read())
            return html.Div([html.Br(),
                             html.Audio(id='myaudio',
                                        src='data:audio/WAV;base64,{}'.format(encoded_image_to_play),
                                        controls=True,
                                        style={"margin-left":"25%"})])


    ####################################################################################
                # disabling div element that is not required
    ####################################################################################
    @_app.callback(
        Output('prediction-audio-upload', 'style'),
        [Input('clustergram-tabs', 'value')])
    def disable_ftp_graph_on_selecting_tabs_audio_upload(value):
        """
        """
        if value == "what-is" or value == "graph":
            return {"display":"none"}


    ####################################################################################
                # disabling div element that is not required
    ####################################################################################
    @_app.callback(Output('display-all-directory-on-status-selection-div', 'style'),
            [Input('clustergram-datasets1', 'value')])
    def disabling_directory_change_filter_type(value):
        """
        Disabling the button after its being clicked once
        """
        if value == "devid" or value == "none" or value == "timestamps":
            return {'display':"none"}


    ####################################################################################
                # disabling div element that is not required
    ####################################################################################
    @_app.callback(Output('display-directory-on-status-selection', 'style'),
            [Input('clustergram-datasets1', 'value')])
    def disabling_directory_on_status_selection(value):
        """
        Disabling the button after its being clicked once
        """
        if value == "devid" or value == "none" or value == "timestamps":
            return {'display':"none"}


    ####################################################################################
                # Displays button to run predictions on selection of file
    ####################################################################################
    @_app.callback(
        Output('display-selection-buttons-single-monitor', 'children'),
        [Input('display-all-directory-on-status-selection-datatable', 'data'),
         Input('display-all-directory-on-status-selection-datatable', 'columns'),
         Input("display-all-directory-on-status-selection-datatable", "derived_virtual_selected_rows")])
    def display_buttons_on_file_selection(rows, columns, indices):
        """
         Displays button on selection of file
        """
        global batch_ftp_file_df
        if indices is not None and indices != []:
            batch_ftp_file_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
            batch_ftp_file_df = batch_ftp_file_df.iloc[indices]
            print batch_ftp_file_df
            return  html.Div(id='selected-files-input-button',
                             children=[
                                 html.Div(className='app-controls-name', children=[
                                     html.Br(),
                                     html.Button(
                                         id='selected-file-selection-button',
                                         children='Run Predictions on selected Files',
                                         n_clicks=0,
                                         n_clicks_timestamp=0,
                                         style={'border':'1px solid #FFBF01', "color":"white"})])],
                             style={"marginTop":"10px"}),


        else:
            batch_ftp_file_df = pd.DataFrame()



    ####################################################################################
                # Displays all the directories of selected status 
    ####################################################################################
    @_app.callback(
        Output('display-directory-on-status-selection', 'children'),
        [Input('display-all-directories-datatable', 'data'),
         Input('display-all-directories-datatable', 'columns'),
         Input("display-all-directories-datatable", "derived_virtual_selected_rows")])
    def display_buttons_predition(rows, columns, indices):
        """
         Displays all the directories of selected status
        """
        if indices is not None and indices != []:
            pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
            directory_selected = pred_df.iloc[indices]["Directories"].tolist()[0]
            print directory_selected
            if directory_selected:
                connect(ftp_path+"/"+directory_selected+"/")
                print ftp.pwd()
                list_all_wavfiles = ftp.nlst("*.wav")
                sort_list_of_all_wavfiles = sort_on_filenames(list_all_wavfiles)
                DATAFRAME_DEVICE_STATUS = pd.DataFrame()
                DATAFRAME_DEVICE_STATUS["File Name"] = sort_list_of_all_wavfiles
            return html.Div(id="display-all-files-on-directory-selection-div",
                            children=[
                                html.Br(),
                                html.Br(),
                                html.P("Select Files: "+ directory_selected + " Device (s)", style={"color":"#FFBF01"}),
                                html.Br(),
                                dash_table.DataTable(id='display-all-directory-on-status-selection-datatable',
                                                     columns=[{"name": i,
                                                               "id": i,}
                                                              for i in DATAFRAME_DEVICE_STATUS.columns],
                                                     data=DATAFRAME_DEVICE_STATUS.to_dict("rows"),
                                                     row_selectable="multi",
                                                     style_table={"Height":"500px",
                                                                  "Width" :"500px"},
                                                     style_header={'backgroundColor': 'rgb(30, 30, 30)', "fontWeight": "bold", 'border': '1px solid white'},
                                                     style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                                 'color': 'white'},),])



    ####################################################################################
                # callback for upload of single file
    ####################################################################################
    @_app.callback(Output(component_id='graph-output-any', component_property='children'),
                   [Input(component_id='upload-data', component_property='contents')],
                   [State('upload-data', 'filename'),
                    State('upload-data', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        """
        check for upload of the files
        """
        if list_of_names:
            if len(list_of_names) == 1:
                if list_of_contents is not None:
                    children = [
                        parse_contents(c, n, d) for c, n, d in
                        zip(list_of_contents, list_of_names, list_of_dates)]
                    return children
            else:
                print "len of files: ", (list_of_names)
                return parse_contents_batch(list_of_contents, list_of_names, list_of_dates)


    ####################################################################################
                    # callback for moniting the drectory with soi
    ####################################################################################
    @_app.callback(Output(component_id='graph-output-graph-tab-do-nothing', component_property='style'),
                  [Input(component_id='display-all-directories-datatable-graph-tab', component_property='data'),
                   Input('display-all-directories-datatable-graph-tab', 'columns'),
                   Input('display-all-directories-datatable-graph-tab', 'derived_virtual_selected_rows'),
                   Input("sound-labels-dropdown-alert-tab", "value"),
                   Input("time-interval-labels-dropdown-alert-tab", "value"),
                   Input("selected-all-active-devices-graph-tab", "n_clicks"),
                   Input("Selected-devices-button-graph-tab", "n_clicks"),
                   Input("phone-number-alert-tab", "value")])
    def update_output_soi_alert(rows,
                                columns,
                                indices,
                                value_soi,
                                value_interval,
                                all_act_dev_clicks,
                                select_dev_clicks,
                                phoneno):
        """
        check for upload of the files
        """
        global  SOI
        global GLOBAL_STOP_THREAD, directory_threads, t
        if indices is not None and indices != [] and select_dev_clicks >= 1 and not GLOBAL_STOP_THREAD:
            all_directories = pd.DataFrame(rows, columns=[c['name'] for c in columns])
            selected_directories = all_directories.iloc[indices]["Directories"].tolist()
            print selected_directories
            if value_soi:
                SOI = [i + " Sounds" for i in value_soi]
            else:
                pass
            if value_interval:
                if value_interval == "all_event":
                    VALUE_INTERVAL = 0
                elif value_interval == "5_mins":
                    VALUE_INTERVAL = 300
            else:
                pass
            if phoneno:
                pass
            else:
                phoneno = None
            directory_threads = []
            ftp_objs = [FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD) for i in range(len(selected_directories))]
            for index, each_dir in enumerate(selected_directories):
                print each_dir
                if index == 0:
                    try:
                        t = threading.Thread(target=directory_procedure_threading,
                                             args=([each_dir], each_dir.split("_")[0][3:],
                                                   ftp_objs[index],
                                                   phoneno))
                        directory_threads.append(t)
                        t.start()
                    except (KeyboardInterrupt, SystemExit):
                        sys.exit(1)

                else:
                    try:
                        t = threading.Thread(target=directory_procedure_threading,
                                             args=([each_dir],
                                                   each_dir.split("_")[0][3:],
                                                   ftp_objs[index],
                                                   phoneno))
                        directory_threads.append(t)
                        t.start()
                    except (KeyboardInterrupt, SystemExit):
                        sys.exit(1)



    ####################################################################################
            # Refreshs continuously the div element to display predictions
    ####################################################################################
    @_app.callback(Output(component_id='graph-output-graph-tab', component_property='children'),
                   [Input("selected-all-active-devices-graph-tab", "n_clicks"),
                    Input("Selected-devices-button-graph-tab", "n_clicks"),
                    Input("sound-labels-dropdown-alert-tab", "value"),
                    Input("interval-component", "n_intervals")])
    def refresh_continuously_display_soi_csv(all_act_dev_clicks, select_dev_clicks, soi_list, n_intervals):
        """
        Refreshs continuously the div element to display predictions
        """
        if all_act_dev_clicks >= 1:
            if os.path.exists("downloaded_audio_files/soi_csv_file.csv"):
                dataframe = pd.read_csv("downloaded_audio_files/soi_csv_file_active_device.csv")
                return html.Div([html.Hr(),
                                 dash_table.DataTable(id='datatable-interactivity-predictions-graph-tab',
                                                      columns=[{"name": i,
                                                                "id": i,
                                                                "deletable": True} for i in dataframe.columns],
                                                      data=dataframe.to_dict("rows"),
                                                      style_table={"Height":"400px",
                                                                   "Width" :"450px"},
                                                      style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                                    "fontWeight": "bold",
                                                                    'border': '1px solid white'},
                                                      style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                                  'color': 'white'},
                                                      row_selectable="single",
                                                      selected_rows=[]),
                                 html.Hr(),
                                 html.Br(),
                                 html.Button(id='button-to-stop-threads',
                                             children='Stop Monitoring',
                                             n_clicks=0,
                                             n_clicks_timestamp=0,
                                             style={'border':'1px solid #FFBF01',
                                                    "margin-top":"20px",
                                                    "margin-left":"20px",
                                                    "color":"white"})])
            else:
                html.Div([html.P("Waiting For getting SOI", style={"textAlign":"center"})])
        else:
            pass
        if select_dev_clicks >= 1:
            if os.path.exists("downloaded_audio_files/soi_csv_file.csv"):
                dataframe = pd.read_csv("downloaded_audio_files/soi_csv_file.csv")
                dataframe["No."] = range(1, dataframe.shape[0]+1)
                devids = []
                for each_ in dataframe["Directory"]:
                    devids.append(get_devid_from_dir(each_))
                del dataframe["Directory"]
                dataframe["DeviceID"] = devids
                dataframe = dataframe[["No.",
                                       "DeviceID",
                                       "Filename",
                                       "Motor",
                                       "Explosion",
                                       "Human",
                                       "Nature",
                                       "Domestic",
                                       "Tools"]]
                if soi_list:
                    color_coding = get_colored_for_soi_columns(soi_list)
                else:
                    color_coding = []
                return html.Div([html.Hr(),
                                 dash_table.DataTable(id='datatable-interactivity-predictions-graph-tab',
                                                      columns=[{"name": i,
                                                                "id": i,
                                                                "deletable": True} for i in dataframe.columns],
                                                      data=dataframe.to_dict("rows"),
                                                      style_table={"height":"300px",
                                                                   'overflowY': 'scroll',
                                                                   "Width" :"450px"},
                                                      style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                                    "fontWeight": "bold",
                                                                    'border': '1px solid white'},
                                                      style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                                                  'color': 'white'},
                                                      style_data_conditional=color_coding,
                                                      selected_rows=[]),
                                 html.Hr(),
                                 html.Br(),
                                 html.Button(id='button-to-stop-threads',
                                             children='Stop Monitoring',
                                             n_clicks=0,
                                             n_clicks_timestamp=0,
                                             style={'border':'1px solid #FFBF01',
                                                    "margin-top":"20px",
                                                    "margin-left":"20px",
                                                    "color":"white"})])
            else:
                return html.Div([html.P("Waiting For getting SOI", style={"textAlign":"center"})])            



    ####################################################################################
                    # callback to stop monitoring
    ####################################################################################
    @_app.callback(Output(component_id='button-to-stop-threads', component_property='disable'),
                   [Input("button-to-stop-threads", "n_clicks")])
    def stop_threads_on_button_click(n_clicks):
        """
        Disabling div elements that are not required
        """
        if n_clicks >= 1:
            global GLOBAL_STOP_THREAD, directory_threads, t
            GLOBAL_STOP_THREAD = True
            time.sleep(5)
            if directory_threads:
                for each in directory_threads:
                    each.join()
            else:
                t.join()
            if os.path.exists("downloaded_audio_files/soi_csv_file.csv"):
                os.remove("downloaded_audio_files/soi_csv_file.csv")
            else:
                pass
            return  True



    ####################################################################################
                    # Disabling div elements that are not required
    ####################################################################################
    @_app.callback(
        Output('button-stop-threads', 'style'),
        [Input('clustergram-tabs', 'value')])
    def disable_ftp_graph_on_selecting_tabs_(value):
        """
        Disabling div elements that are not required
        """
        if value == "datasets" or value == "what-is":
            return {"display":"none"}



    ####################################################################################
                    #  Batch processing of the ftp files
    ####################################################################################
    @_app.callback(
        Output('graph-output-ftp', 'children'),
        [Input('selected-files-input-button', 'n_clicks')])
    def batch_downloading_and_predict(n_clicks):
        """
        Downloading the selected batch of files and processing them to model
        """
        if n_clicks >= 1:
            global display_upload_graph
            emb = []
            malformed = []
            dum_df = batch_ftp_file_df.copy()
            for i in dum_df["File Name"].tolist():
                path = "FTP_downloaded/"+i
                if os.path.exists(path):
                    print "path Exists"
                else:
                    print "Downloading and generating embeddings ", i
                    with open(path, 'wb') as file_obj:
                        ftp.retrbinary('RETR '+ i, file_obj.write)
                try:
                    emb.append(generate_before_predict_BR.main(path, 0, 0, 0))
                except ValueError:
                    print "malformed index", dum_df.loc[dum_df["File Name"] == i].index
                    dum_df = dum_df.drop(dum_df.loc[dum_df["File Name"] == i].index)
                    malformed.append(i)
                    os.remove(path)
            dum_df['features'] = emb
            if len(dum_df["File Name"].tolist()) == 1:
                for each_file in zip(dum_df["File Name"].tolist(), dum_df['features'].values.tolist()):
                    pred_prob, pred = predictions_from_models(each_file[0], each_file[1])
                dum_df["Motor"] = pred_prob[0]
                dum_df["Explosion"] = pred_prob[1]
                dum_df["Human"] = pred_prob[2]
                dum_df["Nature"] = pred_prob[3]
                dum_df["Domestic"] = pred_prob[4]
                dum_df["Tools"] = pred_prob[5]
                display_upload_graph = False
                return table_for_ftp_batch_files(dum_df[dum_df.drop("features", axis=1).columns], malformed)

            elif len(dum_df["File Name"] > 1):
                whole_pred_prob = []
                whole_pred = []
                for each_file in zip(dum_df["File Name"].tolist(), dum_df['features'].values.tolist()):
                    pred_prob, pred = predictions_from_models(each_file[0], each_file[1])
                    whole_pred_prob.append(pred_prob)
                    whole_pred.append(pred)
                whole_pred_prob = np.array(whole_pred_prob)
                whole_pred = np.array(whole_pred)
                print [i for i in whole_pred_prob[:, 0].tolist()]
                dum_df["Motor"] = [i for i in whole_pred_prob[:, 0].tolist()]
                dum_df["Explosion"] = [i for i in whole_pred_prob[:, 1].tolist()]
                dum_df["Human"] = [i for i in whole_pred_prob[:, 2].tolist()]
                dum_df["Nature"] = [i for i in whole_pred_prob[:, 3].tolist()]
                dum_df["Domestic"] = [i for i in whole_pred_prob[:, 4].tolist()]
                dum_df["Tools"] = [i for i in whole_pred_prob[:, 5].tolist()]
                return table_for_ftp_batch_files(dum_df[dum_df.drop("features", axis=1).columns],
                                                 malformed)

            else:
                return html.Div([html.H3("Something went Wrong, Try again",
                                         style={"color":"white"}),
                                 html.P("Note: If problem still persists file might be "+
                                        "corrupted or Input a valid 10 second .wav file",
                                        style={"color":"white"})])




    ####################################################################################
                # Disbaling div elements that are not required
    ####################################################################################
    @_app.callback(
        Output('graph-output-ftp', 'style'),
        [Input('clustergram-tabs', 'value')])
    def disable_ftp_graph_on_selecting_tabs(value):
        """
        Disbaling div elements that are not required
        """
        if value == "datasets" or value == "graph":
            return {"display":"none"}

    @_app.callback(
        Output('graph-output-any', 'style'),
        [Input('clustergram-tabs', 'value')])
    def disable_home_graph_on_selecting_tabs(value):
        """
        Disbaling div elements that are not required
        """
        if value == "what-is" or value == "graph":
            return {"display":"none"}
    @_app.callback(
        Output('graph-output-graph-tab', 'style'),
        [Input('clustergram-tabs', 'value')])
    def disable_graph_graph_on_selecting_tabs(value):
        """
        Disbaling div elements that are not required
        """
        if value == "what-is" or value == "datasets":
            return {"display":"none"}


####################################################################################
        # only declare app/server if the file is being run directly
####################################################################################
if 'DASH_PATH_ROUTING' in os.environ or __name__ == '__main__':
    app = run_standalone_app(layout, callbacks, header_colors, __file__)



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
