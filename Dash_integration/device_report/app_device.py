# -*- coding: utf-8 -*-

from io import BytesIO
from ssl import SSLSocket
import os
import csv
import threading
import argparse
import operator
import ftplib
from datetime import datetime
from ftplib import FTP
import socket
import struct
from datetime import timedelta
import urllib.request
import urllib.parse
import urllib.error
from time import strptime
import glob
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import dash.dependencies
import plotly.graph_objs as go
from .pages import overview
from .utils import Header, make_dash_table
import pathlib


###############################################################################
# get relative data folder
###############################################################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
if os.path.exists("data_downloaded/"):
    pass
else:
    os.mkdir("data_downloaded/")


###############################################################################
# Inputs Required: Path FTP
###############################################################################
PRIMARY_PATH = "/home/user-u0xzU/BNP/"
DIR_REQ = "BNP/"
MAPBOX_ACCESS_TOKEN = "*************"


###############################################################################
# Inputs Required: FTP credentials
###############################################################################
FTP_USERNAME = "user-u0xzU"
FTP_PASSWORD = "**********"
FTP_HOST = '34.211.117.196'


###############################################################################
# READING ARGUMENTS
###############################################################################
DESCRIPTION = "Wildly Acoutsic Monitoring Device Report"
HELP = "Give the Required Arguments"


###############################################################################
# parse the input arguments given from command line
###############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-transmission_mode_csv_path', '--transmission_mode_csv_path', action='store',
                    help=HELP, default="data_downloaded")
PARSER.add_argument('-record_mode_csv_path', '--record_mode_csv_path', action='store',
                    help='Input the path')
RESULT = PARSER.parse_args()


###############################################################################

###############################################################################
if RESULT.transmission_mode_csv_path:
    CSV_FILES = glob.glob(RESULT.transmission_mode_csv_path + "/*.csv")
else:
    pass
if RESULT.record_mode_csv_path:
    CSV_FILES_1 = glob.glob(RESULT.record_mode_csv_path + "/*.csv")
else:
    pass


###############################################################################

###############################################################################
def get_dictionary(csv_files):
    """
    Returns the dictionary with all the csv files as device names
    """
    req_dict = dict()
    for index, each_file in enumerate(csv_files):
        req_dict["Device_" + str(index)] = each_file
    return req_dict


###############################################################################

###############################################################################
DATAFRAME_DEVICE = pd.DataFrame()


###############################################################################
# Main APP LAYOUT
###############################################################################

app = dash.Dash()
server = app.server
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


###############################################################################
# BATTERY PERFORMANCE GRAPH HELPER FUNC TAB
###############################################################################

def SetColor(x):
    """
    coloring scatter plots based on Network quality
    """
    if x > 20:
        return "green"
    elif x > 15 and x <= 20:
        return "yellow"
    elif x >= 10 and x <= 15:
        return "red"
    elif x < 10:
        return "red"


def plot_function(dataframe):
    """
    Data results for file from Transmission mode
    """
    try:
        val = dataframe[["Filename", "DeviceID", "Network_Status", "Time_Stamp", "Network_Type"]].values
        values_list = []
        for each in val:
            values_list.append("Name: " + each[0] + ", Dev: " + each[1] + ", Net_Stat: " + str(each[2]) + " Time: " + each[3] + ", Net_type: " + each[4])

        data = go.Scatter(x=pd.Series(list(range(0, 20000, 1))),
                          text=values_list,
                          mode="markers",
                          hoverinfo="text",
                          name="Network Quality",
                          marker=dict(color=list(map(SetColor, dataframe['Network_Status']))),
                          y=dataframe['Battery_Percentage'])
        return data
    except TypeError:
        val = dataframe[["Filename", "DeviceID"]].values
        values_list = []
        for each in val:
            values_list.append("Name: " + each[0] + ", Dev: " + each[1])

        data = go.Scatter(x=pd.Series(list(range(0, 20000, 1))),
                          text=values_list,
                          mode="markers",
                          name="Network Quality",
                          hoverinfo="text",
                          marker={"color": "blue"},
                          y=dataframe['Battery_Percentage'])
        return data


###############################################################################
# Reads csv file and returns pandas dataframe
###############################################################################
def read_csv_file(csv_file_name):
    """
    Return dataframe
    """
    dataframe = pd.read_csv(csv_file_name)
    return dataframe


###############################################################################
# Different dataframe with filter based on date and time
###############################################################################
def get_data_specific_date(dataframe):
    """
    separates data and time stamp
    """
    list_of_date_time = dataframe['Time_Stamp'].values.tolist()
    only_date = []
    date_value = []
    for each in list_of_date_time:
        date, value = each.split("-")[0], each.split("-")[0].split("/")[0]
        only_date.append(date)
        date_value.append(value)
    dataframe["Date"] = only_date
    return dataframe, list(set(date_value))


def get_data_specific_time(dataframe):
    """
    separates time from time stamp
    """
    list_of_date_time = dataframe["Time_Stamp"].values.tolist()
    only_time = []
    timeFormat = "%H:%M:%S"
    for each in list_of_date_time:
        time = str(datetime.strptime(each.split("-")[1], timeFormat)).split(" ")[1]
        only_time.append(time)

    dataframe["Time"] = only_time
    return dataframe


def filter_on_date(dataframe, date):
    """
    Filtering only specific date
    """
    return dataframe.loc[dataframe["Date"].apply(lambda arr: arr.split("/")[0] == date[0] or arr.split("/")[0] == date[1])]


def get_dataframe_for_plotting_transmission(file_index):
    """
    Returns Data frame more suitable format for plotting tramssion mode files
    """
    dataframe = read_csv_file(file_index)
    df, date_values = get_data_specific_date(dataframe)
    df_date = filter_on_date(df, date_values)
    df_date = get_data_specific_time(df_date)
    return df_date


def get_dataframe_for_plotting_recording(file_index):
    """
    Returns Data frame more suitable format for plotting recording mode files
    """
    dataframe = read_csv_file(file_index)
    df, date_values = get_data_specific_date(dataframe)
    df_date = filter_on_date(df, date_values)
    df_date = get_data_specific_time(df_date)
    return df_date


###############################################################################
# CALLBACK FOR URL INPUT: DEVICE TRANSMISSION
###############################################################################

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    """
    Condition check for menu selection
    """
    if pathname == "/acoustic-device-report/transmission-performance":
        return html.Div(
            [Header(app),
             html.Div(
                 [html.Div(
                     [html.Div(
                         [html.H6(["Select Devices"], className="subtitle padded"),
                          html.Div(
                              [html.Div(
                                  [dash_table.DataTable(id='datatable-interactivity-transmission',
                                                        columns=[{"name": i,
                                                                  "id": i,
                                                                  "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                        data=DATAFRAME_DEVICE.to_dict("rows"),
                                                        row_selectable="multi",
                                                        style_table={"maxHeight": "200px",
                                                                     "maxWidth": "200px"})])
                               ], style={"margin-bottom": "10px"})
                          ], className="six columns")
                      ], className="row "),
                  html.Div(
                      [html.Div(id="inside-transmission")],
                      className="row "),
                  ], className="sub_page"),
             ], className="page")

    ###########################################################################
    # CALLBACK FOR URL INPUT: LOCATION
    ###########################################################################
    elif pathname == "/acoustic-device-report/location-details":

        return html.Div(
            [Header(app),
             html.Div(
                 [html.Div(
                     [html.Div(
                         [html.H6(["Select Devices"], className="subtitle padded"),
                          html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-location',
                                                                   columns=[{"name": i,
                                                                             "id": i,
                                                                             "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                   data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                   row_selectable="multi",
                                                                   style_table={"maxHeight": "200px",
                                                                                "maxWidth": "200px"})
                                              ]),
                                    ], style={"margin-bottom": "10px"})
                          ], className="six columns")
                      ], className="row "),
                  html.Div(
                      [html.Div(id="inside-location")],
                      className="row ")
                  ], className="sub_page")
             ], className="page")

    ###########################################################################
    # CALLBACK FOR URL INPUT: BATTERY PERFORMANCE
    ###########################################################################
    elif pathname == "/acoustic-device-report/battery-performance":
        return html.Div(
            [Header(app),
             html.Div(
                 [html.Div(
                     [html.Div(
                         [html.H6(["Select Devices"], className="subtitle padded"),
                          html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-battery',
                                                                   columns=[{"name": i,
                                                                             "id": i,
                                                                             "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                   data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                   row_selectable="multi",
                                                                   style_table={"maxHeight": "200px",
                                                                                "maxWidth": "200px"})
                                              ]),
                                    ], style={"margin-bottom": "10px"})
                          ], className="six columns")
                      ], className="row "),
                  html.Div(
                      [html.Div(id="inside-battery")],
                      className="row "),
                  html.Div(
                      [html.A(id="my-link")],
                      className="row ")
                  ], className="sub_page")
             ], className="page")

    ###########################################################################
    # CALLBACK FOR URL INPUT: DEVICE DETAILS
    ###########################################################################
    elif pathname == "/acoustic-device-report/device-details":
        return html.Div([Header(app),
                         html.Div(
                             [html.Div(
                                 [html.Div(
                                     [html.H6(["Select Devices"], className="subtitle padded"),
                                      html.Div(
                                          [html.Div([dash_table.DataTable(id='datatable-interactivity-device-details',
                                                                          columns=[{"name": i,
                                                                                    "id": i,
                                                                                    "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                          data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                          row_selectable="single",
                                                                          style_table={"maxHeight": "200px",
                                                                                       "maxWidth": "200px"})])
                                           ], style={"margin-bottom": "10px"})
                                      ], className="six columns"),
                                  html.Div(id="device-details-id", className="six columns")
                                  ], className="row "),

                              # Row 2
                              html.Div(
                                  [html.Div(id="inside-device-details")],
                                  className="row ")
                              ], className="sub_page")], className="page")

    ###########################################################################
    # CALLBACK FOR URL INPUT: REVIEWS
    ###########################################################################
    elif pathname == "/acoustic-device-report/reviews":
        return html.Div(
            [Header(app),
             html.Div(
                 [html.Div(
                     [html.Div(
                         [html.H6(["Select Devices"], className="subtitle padded"),
                          html.Div(
                              [html.Div(
                                  [dash_table.DataTable(id='datatable-interactivity-device-details',
                                                        columns=[{"name": i,
                                                                  "id": i,
                                                                  "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                        data=DATAFRAME_DEVICE.to_dict("rows"),
                                                        row_selectable="single",
                                                        style_table={"maxHeight": "200px",
                                                                     "maxWidth": "200px"})
                                   ])
                               ], style={"margin-bottom": "10px"})
                          ], className="six columns"),
                      html.Div(
                          [html.Div(
                              [html.H6("Device Transmission Performance Review",
                                       className="subtitle padded",
                                       style={"margin-left": "5%"}),
                               html.Br([]),
                               html.Div(
                                   [html.P("Start Time of the Device:    17:59:03 - 19/07/2019",
                                           style={"margin-left": "5%"}),
                                    html.P("Estimated Recorded Files:   3600 (.wav) Files. ",
                                           style={"margin-left": "5%"})],
                                   style={"color": "#7a7a7a"})
                               ], className="six columns"),

                           html.Div(
                               [html.H6("Network Related", className="subtitle padded"),
                                html.Br([]),
                                html.Div(
                                    [html.P("TYpe of Network Quality"),
                                     html.P("Consistency of the Network Quality")],
                                    style={"color": "#7a7a7a"})
                                ],
                               style={"margin-top": "5%",
                                      "margin-right": "-20%"},
                               className="six columns"),

                           html.Div(
                               [html.H6("Summary", className="subtitle padded"),
                                html.Br([]),
                                html.Div(
                                    [html.P(""),
                                     html.P("")],
                                    style={"color": "#7a7a7a"})
                                ],
                               style={"margin-top": "5%"},
                               className="six columns")
                           ],
                          className="row ")
                      ],
                     className="row "),

                  html.Div([html.Div(id="inside-device-details")],
                           className="row ")],
                 className="sub_page")
             ], className="page")

    ###########################################################################
    # CALLBACK FOR URL INPUT: FULL VIEW
    ###########################################################################
    elif pathname == "/acoustic-device-report/full-view":
        return overview.create_layout(app)

    ###########################################################################
    # CALLBACK FOR URL INPUT: OVERVIEW
    ###########################################################################
    elif pathname == "/acoustic-device-report/overview":
        return overview.create_layout(app)

    ###########################################################################
    # CALLBACK FOR URL INPUT: DEFAULT / HOME PAGE
    ###########################################################################
    else:
        connect_group(PRIMARY_PATH)
        list_wavfiles = True
        dir_n_timestamp, directories_time_list = last_ftp_time(PRIMARY_PATH)
        dir_n_timestamp, status = active_or_inactive(dir_n_timestamp, directories_time_list)
        list_device = [each[0] for each in dir_n_timestamp]
        list_timestamps = [each[1] + " [IST]" for each in dir_n_timestamp]
        directory_threads = []
        for index, each_dir in enumerate(list_device):
            if index == 0:
                threads = threading.Thread(target=write_csv, args=("data_downloaded/" + each_dir + ".csv", DIR_REQ + each_dir))
                directory_threads.append(threads)
                threads.start()
            else:
                threads = threading.Thread(target=write_csv, args=("data_downloaded/" + each_dir + ".csv", DIR_REQ + each_dir))
                directory_threads.append(threads)
                threads.start()
        DATAFRAME_DEVICE_ACTIVE = pd.DataFrame()
        DATAFRAME_DEVICE_ACTIVE["Device ID"] = list_device
        display_device_list = []
        if list_device:
            for each_value_ in list_device:
                if os.path.exists("data_downloaded/" + each_value_ + ".csv"):
                    display_device_list.append(each_value_)
                else:
                    pass

        DATAFRAME_DEVICE["Select Device"] = display_device_list
        DATAFRAME_DEVICE_ACTIVE["Last Modified Time"] = list_timestamps
        DATAFRAME_DEVICE_ACTIVE['Report'] = ["Download Report"] * DATAFRAME_DEVICE_ACTIVE.shape[0]
        DATAFRAME_DEVICE_ACTIVE["Status   (5 mins)"] = status
        DATAFRAME_DEVICE_ACTIVE = DATAFRAME_DEVICE_ACTIVE.sort_values(by=['Last Modified Time'], ascending=False)
        DATAFRAME_DEVICE_ACTIVE["Device No."] = list(range(1, len(list_device) + 1))
        DATAFRAME_DEVICE_ACTIVE = DATAFRAME_DEVICE_ACTIVE[["Device No.", "Device ID", "Last Modified Time", "Report", "Status   (5 mins)"]]

        #######################################################################
        # Return the page with Directory status and Graph
        #######################################################################
        fig_active = get_figure_active(list_device, status)
        if list_wavfiles:
            return html.Div(
                [
                    html.Div([Header(app)]),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [html.H5("Device (s) Summary", style={"text-align": "center", 'text-decoration': 'underline'}),
                                         html.Div([
                                             html.Br([]),
                                             html.Br([]),
                                             html.P("Active Device (s) : " + str(len(list_device)) + " Device (s)",
                                                    style={'text-decoration': 'underline',
                                                           "margin-left": "10px"}
                                                    ),
                                             html.Br([]),
                                             html.Div([Table(DATAFRAME_DEVICE_ACTIVE, "Report")],
                                                      style={"padding-left": "2px",
                                                             "padding-right": "8px",
                                                             "margin-top": "20px"}
                                                      )
                                         ],
                                            style={"margin-bottom": "10px"}),
                                            html.Br([]),
                                            html.Br([]),
                                            html.P("NOTE: Refresh to see recent activity.[Needs Refresh]",
                                                   style={"margin-left": "70%", 'text-decoration': 'underline'})

                                         ],
                                        className="product",
                                    )
                                ],
                                className="row",
                            ),
                            # Row
                            html.Div(
                                [html.Div([
                                    html.Div([
                                        html.Div([
                                            html.H6("Device Location Plot",
                                                    className="subtitle padded",
                                                    style={"margin-bottom": "20px"}),
                                            dcc.Graph(figure={"data": fig_active["data"],
                                                              "layout":fig_active["layout"]},
                                                      config={"displayModeBar": False})
                                        ],
                                            className="twelve columns",
                                            style={"margin-left": "-57%",
                                                   "margin-top": "20%"})
                                    ],
                                        style={"margin-bottom": "10px"})
                                ],
                                    className="six columns"),
                                    html.Div(id="device-details-id",
                                             className="six columns")
                                ], className="row ")
                        ],
                        className="sub_page")
                ],
                className="page")


###############################################################################
# connect to FTP server
###############################################################################
def connect(primary_path):
    '''
    To connect to ftp
    '''
    global ftp
    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
    print("connected to FTP")
    ftp.cwd(primary_path)


def connect_group(primary_path):
    """
    To connect to ftp server for with different object
    """
    global ftp_group
    ftp_group = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
    ftp_group.cwd(primary_path)


###############################################################################
# check for wav files only in FTP server
###############################################################################
def check_wav_files():
    """
    Checks only wav files and returns the list
    """
    try:
        wavfiles_list = ftp.nlst("*.wav")
    except:
        return None
    return wavfiles_list


###############################################################################
# sort files based on names
###############################################################################
def sort_on_filenames(files_list):
    """
    Sorts the ftp files based on names: Ascending order
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
        wav_files_toint = list(map(int, wav_files_number))
        sorted_wavfiles = sorted(wav_files_toint)

        for sorted_file in sorted_wavfiles:
            wav_files_list.append(character + str(sorted_file)[:8] + "_" + str(sorted_file)[8:] + extension)
        return wav_files_list
    else:
        return []


###############################################################################
# sort files based on ftp timestamps: Very slow for large files
###############################################################################
def get_list_files_sorted(ftp_path):
    """
    sort files based on ftp timestamps
    """
    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
    ftp.cwd(ftp_path)
    wav_files = ftp.nlst()
    dictionary = {}
    for name in wav_files[:1000]:
        # try:
        if (name[-3:] == 'wav') or (name[-3:] == 'WAV'):
            time1 = ftp.voidcmd("MDTM " + name)
            dictionary[time1[4:]] = name
    sorted_wav_files_list = sorted(list(dictionary.items()), key=operator.itemgetter(0))
    return sorted_wav_files_list


###############################################################################
# Removes duplicate files from iteration
###############################################################################
def get_without_duplicates(csv_filename, sorted_wavfiles):
    """
    Removes duplicate files from iteration
    """
    if os.path.exists(csv_filename):
        dataframe = pd.read_csv(csv_filename)["Filename"].values.tolist()
        if dataframe:
            non_repeated = []
            for each_value in sorted_wavfiles:
                if each_value not in dataframe:
                    non_repeated.append(each_value)
                else:
                    pass
            return non_repeated
        return sorted_wavfiles
    else:
        return sorted_wavfiles


###############################################################################
# Create and write csv file with all the details
###############################################################################
def write_csv(csv_filename, ftp_path):
    """
    Create and write csv file with all the details
    """
    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
    ftp.cwd(ftp_path)
    wav_files = ftp.nlst()
    sorted_files = sort_on_filenames(wav_files)

    ###########################################################################
    # CSV file column details
    ###########################################################################
    non_repeated = get_without_duplicates(csv_filename, sorted_files)
    wav_info_tags = ["Filename", "Operator", "DeviceID", "Battery_Voltage", "Battery_Percentage",\
                     "Network_Status", "Network_Type", "Firmare_Revision", "Time_Stamp", "Latitude", "Longitude", \
                     "Clock", "ChunkID", "TotalSize", "Format", "SubChunk1ID", "SubChunk1Size", "AudioFormat", \
                     "NumChannels", "SampleRate", "ByteRate", "BlockAlign", "BitsPerSample", "SubChunk2ID", \
                     "SubChunk2Size"]
    wav_info_tags2 = ["ChunkID", "TotalSize", "Format", "SubChunk1ID", "SubChunk1Size", \
                      "AudioFormat", "NumChannels", "SampleRate", "ByteRate", "BlockAlign", \
                      "BitsPerSample", "SubChunk2ID", "SubChunk2Size"]

    ###########################################################################
    # If CSV file already exists append rows to it
    ###########################################################################
    if os.path.exists(csv_filename):
        with open(csv_filename, "a") as file_object:
            wav_information_object = csv.writer(file_object)
            file_object.flush()
            for wav_file in non_repeated:
                try:
                    wav_header, extra_header = get_wavheader_extraheader(wav_file, ftp_path)
                    information_value = [wav_file]
                    for index_value, each_tag_value in enumerate(extra_header):
                        try:
                            corresponding_tag, corresponding_value = each_tag_value.split(":")
                            if corresponding_tag == " Signal":
                                corresponding_value = corresponding_value.split("-")
                                for signal_split in corresponding_value:
                                    information_value.append(signal_split)
                            else:
                                information_value.append(corresponding_value)

                        except ValueError:
                            corresponding_value = ":".join(each_tag_value.split(":")[1:])
                            information_value.append(corresponding_value)

                    for tag in wav_info_tags2:
                        value = wav_header[tag]
                        information_value.append(value)
                    wav_information_object.writerow(information_value)
                    file_object.flush()
                except socket.error:
                    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
                    ftp.cwd(ftp_path)
                except ftplib.error_temp:
                    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
                    ftp.cwd(ftp_path)
                except struct.error:
                    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
                    ftp.cwd(ftp_path)

    ###########################################################################
                    # If CSV file doesn't exists create new and write
    ###########################################################################
    else:
        with open(csv_filename, "w") as file_object:
            wav_information_object = csv.writer(file_object)
            wav_information_object.writerow(wav_info_tags)
            file_object.flush()
            for wav_file in non_repeated:
                try:
                    wav_header, extra_header = get_wavheader_extraheader(wav_file, ftp_path)
                    information_value = [wav_file]
                    for index_value, each_tag_value in enumerate(extra_header):
                        try:
                            corresponding_tag, corresponding_value = each_tag_value.split(":")
                            if corresponding_tag == " Signal":
                                corresponding_value = corresponding_value.split("-")
                                for signal_split in corresponding_value:
                                    information_value.append(signal_split)
                            else:
                                information_value.append(corresponding_value)

                        except ValueError:
                            corresponding_value = ":".join(each_tag_value.split(":")[1:])
                            information_value.append(corresponding_value)

                    for tag in wav_info_tags2:
                        value = wav_header[tag]
                        information_value.append(value)
                    wav_information_object.writerow(information_value)
                    file_object.flush()
                except socket.error:
                    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
                    ftp.cwd(ftp_path)
                except ftplib.error_temp:
                    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
                    ftp.cwd(ftp_path)
                except struct.error:
                    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
                    ftp.cwd(ftp_path)


###############################################################################
# FTP class to read data without downloading file
###############################################################################
class FtpFile:
    """
    sdfsg
    """

    def __init__(self, ftp, name):
        self.ftp = ftp
        self.name = name
        self.size = 10240
        self.pos = 0

    def seek(self, offset, whence):
        if whence == 0:
            self.pos = offset
        if whence == 1:
            self.pos += offset
        if whence == 2:
            self.pos = self.size + offset

    def tell(self):
        return self.pos

    def read(self, ftp_path, size=None):
        ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
        ftp.cwd(ftp_path)
        if size is None:
            size = self.size - self.pos
        data = B""
        ftp.voidcmd('TYPE I')
        cmd = "RETR {}".format(self.name)
        conn = ftp.transfercmd(cmd, self.pos)
        try:
            while len(data) < size:
                buf = conn.recv(min(size - len(data), 8192))
                if not buf:
                    break
                data += buf
            # shutdown ssl layer (can be removed if not using TLS/SSL)
            if SSLSocket is not None and isinstance(conn, SSLSocket):
                conn.unwrap()
        finally:
            conn.close()
        try:
            ftp.voidresp()
        except:
            pass
        self.pos += len(data)
        return data


###############################################################################
# Get wavheader information
###############################################################################
def get_wavheader_extraheader(name, ftp_path):
    '''
    To read wav file header details
    '''
    wavheader_dict = {}
    ftp = FTP(FTP_HOST, user=FTP_USERNAME, passwd=FTP_PASSWORD)
    ftp.cwd(ftp_path)

    if (name[-3:] == 'wav') or (name[-3:] == 'WAV'):
        try:
            file_header_info = BytesIO(FtpFile(ftp, name).read(ftp_path, 264))

            riff, size, fformat = struct.unpack('<4sI4s', file_header_info.read(12))
            chunkoffset = file_header_info.tell()

            chunk_header = file_header_info.read(8)
            subchunkid, subchunksize = struct.unpack('<4sI', chunk_header)
            chunkoffset = file_header_info.tell()

            aformat, channels, samplerate, byterate, blockalign, bps = struct.unpack('HHIIHH', file_header_info.read(16))
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

            for each_value in zip(wav_header, ["ChunkID", "TotalSize", "Format", "SubChunk1ID", "SubChunk1Size",
                                               "AudioFormat", "NumChannels", "SampleRate", "ByteRate", "BlockAlign",
                                               "BitsPerSample", "SubChunk2ID", "SubChunk2Size"]):
                wavheader_dict[each_value[1]] = each_value[0]

            extra_header_info = extra_header.decode("ascii").split(',')

            return wavheader_dict, extra_header_info
        except UnicodeDecodeError:
            print("Got Unexpected File")
            return None, None


###############################################################################
# Get directory and time stamp
###############################################################################
def get_directory_timestamp_listed():
    """
    Get directory and time stamp
    """
    dir_list = []
    ftp_group.dir(dir_list.append)
    device_list = []
    timestamps = []
    for each_value in dir_list:
        if each_value[:2] == "dr":
            if not each_value.split(" ")[-1] == "CorruptFiles":
                device_list.append(each_value.split(" ")[-1])
                timestamps.append(":".join(each_value.split(" ")[-4:-1]))

    return device_list, timestamps


###############################################################################

###############################################################################
def check_wav_file_size(each_wav_file, blockalign, samplerate):
    """
    Checks for wavfile size and return if it is completely
    uploaded to ftp server or not
    """
    ftp_group.sendcmd("TYPE I")
    if ftp_group.size(each_wav_file) > samplerate * blockalign * 10:
        ftp_group.sendcmd("TYPE A")
        return True
    else:
        ftp_group.sendcmd("TYPE A")
        return False


###############################################################################
# Groups device based on device id
###############################################################################
def group_by_device_id():
    """
    """
    ftp_path = PRIMARY_PATH
    while 1:
        print("Process in BG")
        files_list = ftp_group.nlst()
        try:
            for wav_file in files_list:
                if (wav_file[-3:] == 'wav') or (wav_file[-3:] == 'WAV'):
                    wav_header, extra_header = get_wavheader_extraheader(wav_file, ftp_path)
                    if wav_header:
                        blockalign = wav_header["BlockAlign"]
                        samplerate = wav_header["SampleRate"]
                        for index_value, each_tag_value in enumerate(extra_header):
                            device_id_tag = extra_header[index_value].split(":", 1)[0]
                            if device_id_tag == ' DeviceID':
                                Device_ID = extra_header[index_value].split(":", 1)[1]
                                if check_wav_file_size(wav_file, blockalign, samplerate):
                                    if Device_ID in ftp_group.nlst():
                                        pass
                                    else:
                                        ftp_group.mkd(Device_ID)
                                    ftp_group.rename(PRIMARY_PATH + wav_file, PRIMARY_PATH + Device_ID + '/' + wav_file)
                                else:
                                    pass
                    else:
                        pass
                else:
                    pass
        except socket.error:
            connect_group(PRIMARY_PATH)
        except ftplib.error_temp:
            connect_group(PRIMARY_PATH)
        except struct.error:
            connect_group(PRIMARY_PATH)


###############################################################################
# Creates a table in dash format from pandas dataframe format
###############################################################################
def Table(dataframe, column_name):
    """
    Creates a table in dash format from pandas dataframe format
    column name given will be hyperlinked
    """
    rows = []
    for i in range(len(dataframe)):
        row = []
        for col in dataframe.columns:
            value = dataframe.iloc[i][col]

            if os.path.exists("data_downloaded/" + dataframe.iloc[i][1] + ".csv"):
                csv_string = pd.read_csv("data_downloaded/" + dataframe.iloc[i][1] + ".csv")
                csv_string = csv_string.to_csv(index=False, encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
                if col == column_name:
                    print("Im executed", dataframe.iloc[i][1] + ".csv")
                    cell = html.Td(html.A(id="download-report",
                                          href=csv_string,
                                          download="data_downloaded/" + dataframe.iloc[i][1] + ".csv",
                                          children=[value],
                                          style={"color": "blue", 'text-decoration': 'underline'}),
                                   style={"padding-top": "10px",
                                          "padding-right": "13px",
                                          "padding-left": "10px",
                                          'text-align': 'center', })
                else:
                    cell = html.Td(children=value,
                                   style={"padding-top": "10px",
                                          'color': 'black',
                                          "padding-right": "13px",
                                          "padding-left": "10px",
                                          'text-align': 'center'})
                row.append(cell)
            else:
                if col == column_name:
                    cell = html.Td(html.A(children=[value],
                                          style={"color": "black",
                                                 'text-decoration': 'underline'}),
                                   style={"padding-top": "10px",
                                          "padding-right": "13px",
                                          "padding-left": "10px",
                                          'text-align': 'center'})
                else:
                    cell = html.Td(children=value,
                                   style={"padding-top": "10px",
                                          'color': 'black',
                                          "padding-right": "13px",
                                          "padding-left": "10px",
                                          'text-align': 'center'})
                row.append(cell)

        rows.append(html.Tr(row))
    return html.Table(
        [html.Tr([html.Th(col,
                          style={"padding-top": "10px",
                                 "padding-right": "30px",
                                 'color': 'black',
                                 "padding-left": "30px",
                                 "padding-bottom": '10px',
                                 'text-align': 'center'}) for col in dataframe.columns])] + rows)


###############################################################################
# Map plot for active and inactive devices on home page
###############################################################################
LATITUDES_ACTIVE = []
LONGITUDES_ACTIVE = []
DEVICE_ACTIVE_COLOR = []
DEVICE_LIST_ACTIVE = []


def get_data_active(device_list, color):
    print(DEVICE_ACTIVE_COLOR)
    data = go.Scattermapbox(
        mode='markers',
        lat=LATITUDES_ACTIVE,
        lon=LONGITUDES_ACTIVE,
        text=device_list,
        marker=go.scattermapbox.Marker(size=7, color=color, opacity=0.7),
        subplot='mapbox')
    return data


def get_figure_active(list_of_devices, status_list):
    """
    Active devices / home page maps
    """
    global LATITUDES_ACTIVE, LONGITUDES_ACTIVE, DEVICE_ACTIVE_COLOR, DEVICE_LIST_ACTIVE
    trace = []
    for dev, stat in zip(list_of_devices, status_list):
        if os.path.exists("data_downloaded/" + dev + ".csv"):
            dataframe = read_csv_file("data_downloaded/" + dev + ".csv")
            cord_ = dataframe[["Latitude", "Longitude"]]
            print(cord_.iloc[0][0])
            if cord_.iloc[0][0] and cord_.iloc[0][1]:
                LATITUDES_ACTIVE.append(cord_.iloc[0][0])
                LONGITUDES_ACTIVE.append(cord_.iloc[0][1])
                DEVICE_LIST_ACTIVE.append(dev)
                if stat == "Active":
                    DEVICE_ACTIVE_COLOR.append("green")
                else:
                    DEVICE_ACTIVE_COLOR.append("red")
            else:
                pass
        else:
            pass
    data = get_data_active(DEVICE_LIST_ACTIVE, DEVICE_ACTIVE_COLOR)
    trace.append(data)
    layout = get_layout_active()
    fig = go.FigureWidget(data=trace, layout=layout)
    return fig


def get_layout_active():
    """
    """
    if LATITUDES_ACTIVE:
        layout = go.Layout(height=400,
                           width=720,
                           hovermode="x",
                           margin={"r": 0,
                                   "t": 0,
                                   "b": 0,
                                   "l": 0},

                           mapbox=go.layout.Mapbox(
                               accesstoken=MAPBOX_ACCESS_TOKEN,
                               domain={'x': [0, 1.0], 'y': [0, 1]},
                               bearing=0,
                               center=go.layout.mapbox.Center(
                                   lat=float("{0:.4f}".format(LATITUDES_ACTIVE[0])),
                                   lon=float("{0:.4f}".format(LONGITUDES_ACTIVE[0]))
                               ),
                               pitch=0,
                               style="light",
                               zoom=10)
                           )
        return layout

    else:
        layout = go.Layout(height=400,
                           width=720,
                           hovermode="x",
                           margin={"r": 0,
                                   "t": 0,
                                   "b": 0,
                                   "l": 0},
                           mapbox=go.layout.Mapbox(
                               accesstoken=MAPBOX_ACCESS_TOKEN,
                               domain={'x': [0, 1.0], 'y': [0, 1]},
                               bearing=0,
                               pitch=0,
                               style="light",
                               zoom=10))

        return layout


###############################################################################
# Get all the directory in specified ftp server path
###############################################################################
def directory_details(ftp_path):
    """

    """
    dir_n_timestamp = []
    ftp_group.sendcmd("TYPE A")
    lines = []
    now = datetime.now()
    now_year = now.strftime("%Y")
    datetimeFormat1 = '%Y/%m/%d-%H:%M'
    ftp_group.dir(lines.append)

    for line in lines:
        if line[0] == 'd':
            directory = line.split(' ')[-1]
            if len(line.split(" ")[-2].split(":")) == 2:
                print(line.split(" "))
                if line.split(' ')[-4]:
                    month = line.split(' ')[-4]
                else:
                    month = line.split(' ')[-5]
                timestamp1 = now_year + '/' + str(strptime(month, '%b').tm_mon) + '/' +\
                    line.split(' ')[-3] + '-' + line.split(' ')[-2]
                time2 = str(datetime.strptime(timestamp1, datetimeFormat1) + timedelta(minutes=330))

                dir_n_time = directory, time2, 'active'
                dir_n_timestamp.append(dir_n_time)
                print("timestamp(time):", time2)
            else:
                timestamp1 = line.split(' ')[-2] + '/' + str(strptime(line.split(' ')[-5], '%b').tm_mon)\
                    + '/' + line.split(' ')[-4]
                print("timestamp(year):", timestamp1)
                dir_n_time = directory, timestamp1, 'inactive'
                dir_n_timestamp.append(dir_n_time)

    print("dir_n_timestamp:", dir_n_timestamp)
    return dir_n_timestamp


def last_ftp_time(ftp_path):
    """
    Returns recent activity timestamp of the directory
    """
    datetimeFormat2 = '%Y-%m-%d %H:%M:%S'
    now = datetime.now()
    directories_time_list = []
    timestamp2 = now.strftime(datetimeFormat2)

    dir_n_timestamp = directory_details(ftp_path)
    for dir_n_time in dir_n_timestamp:
        if dir_n_time[2] == 'active':
            if len(dir_n_time[1].split(' ')) != 1:
                time_diff = datetime.strptime(timestamp2, datetimeFormat2) - datetime.strptime(dir_n_time[1], datetimeFormat2)
                directories_time_list.append(str(time_diff))
                print("last_ftp_time_list:", time_diff)
        else:
            directories_time_list.append('inactive')
    return dir_n_timestamp, directories_time_list


###############################################################################
# Return active and inactive devices based on 5 Mins activity
###############################################################################
def active_or_inactive(dir_n_timestamp, directories_time_list):
    """
    Return active and inactive devices based on 5 Mins activity
    """
    status = []
    for td in directories_time_list:
        if td == 'inactive':
            status.append('Inactive')
        else:
            if len(td.split(' ')) == 1:
                hour = int(td.split(":")[0])
                minute = int(td.split(":")[1])
                seconds_ = int(td.split(":")[2])
                seconds = hour * 60 * 60 + minute * 60 + seconds_
            else:
                day = td.split(" ")[0]
                hour = td.split(" ")[2].split(":")[0]
                minute = td.split(" ")[2].split(":")[1]
                seconds_ = td.split(" ")[2].split(":")[2]
                seconds = day * 24 * 60 * 60 + hour * 60 * 60 + minute * 60 + seconds_
            if seconds <= 300:
                status.append('Active')
            else:
                status.append('Inactive')
    return dir_n_timestamp, status


###############################################################################
# TRANSMISSION HELPER FUNCTION
###############################################################################

def plot_function_bar(dataframe):
    """
    Bar plot for transmision file details
    """
    val = dataframe[["Filename", "DeviceID", "Network_Status", "Time_Stamp", "Network_Type"]].values
    values_list = []
    for each in val:
        values_list.append("Name: " + each[0] + ", Dev: " + each[1] + ", Net_Stat: " + str(each[2]) + " Time: " + each[3] + ", Net_type: " + each[4])

    trace1 = go.Bar(
        x=[dataframe["DeviceID"].iloc[0]],
        y=[dataframe.shape[0]],
        name="Duration:- " + str(get_time_difference(dataframe["Time_Stamp"].iloc[0],
                                                     dataframe["Time_Stamp"].iloc[dataframe.shape[0] - 1])),
        text="No. Files:- " + str(dataframe.shape[0]),
        textposition="outside")
    return trace1


###############################################################################
# TIME STAMP DIFFERENCE HELPER FUNC
###############################################################################

def get_time_difference(timestamp1, timestamp2):
    """
    Returns the difference of time stamps
    """
    datetimeFormat = '%Y/%m/%d-%H:%M:%S'
    time_diff = datetime.strptime(timestamp2, datetimeFormat) - datetime.strptime(timestamp1, datetimeFormat)
    print("\ntime_diff :", str(time_diff))
    return time_diff


###############################################################################
# LOCATION HELPER FUNC
###############################################################################


def get_data(name, latitudes, longitudes, device_name_location):
    """
    Returns the data list relative to plotly figure object
    """
    data = go.Scattermapbox(
        mode='markers',
        lat=[latitudes],
        lon=[longitudes],
        name=name,
        text=device_name_location,
        subplot='mapbox'
    )
    return data


def get_layout(latitudes, longitudes):
    """
    return the layout for latitudes and longitues given
    """
    print(latitudes, longitudes)
    layout = go.Layout(height=400,
                       width=690,
                       hovermode='closest',
                       margin={"r": 0,
                               "t": 0,
                               "b": 0,
                               "l": 0},

                       mapbox=go.layout.Mapbox(
                           accesstoken=MAPBOX_ACCESS_TOKEN,
                           domain={'x': [0, 1.0], 'y': [0, 1]},
                           bearing=0,
                           center=go.layout.mapbox.Center(
                               lat=float("{0:.4f}".format(latitudes)),
                               lon=float("{0:.4f}".format(longitudes))),
                           pitch=0,
                           zoom=7))

    return layout


def get_figure(list_of_devices):
    """
    Returns the dash plotly figure object
    """
    trace = []
    for _, each in enumerate(list_of_devices):
        dataframe = read_csv_file("data_downloaded/" + each + ".csv")
        latitudes = dataframe["Latitude"].iloc[2]
        longitudes = dataframe["Longitude"].iloc[2]
        data = get_data(each, latitudes, longitudes, each)
        trace.append(data)
    layout = get_layout(latitudes, longitudes)
    fig = go.FigureWidget(data=trace, layout=layout)
    return fig


###############################################################################
# CALLBACK FOR TRANSMISSION
###############################################################################


@app.callback(
    dash.dependencies.Output("inside-transmission", "children"),
    [Input('datatable-interactivity-transmission', 'data'),
     Input('datatable-interactivity-transmission', 'columns'),
     Input("datatable-interactivity-transmission", "derived_virtual_selected_rows")])
def update_figure_transmission(rows, columns, indices):
    """
    Callback for plots on  Transmission files details
    """
    fig = []
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        selected = pred_df.values.tolist()
        if selected:
            if len(selected) == 1:
                df_date = get_dataframe_for_plotting_transmission("data_downloaded/" + selected[0] + ".csv")
                device1 = plot_function_bar(df_date)
                fig.append(device1)
            else:
                for each_fig in selected:
                    df_date = get_dataframe_for_plotting_transmission("data_downloaded/" + each_fig + ".csv")
                    device1 = plot_function_bar(df_date)
                    fig.append(device1)
            return html.Div([html.H6("Device Transmission Performances",
                                     className="subtitle padded"),
                             dcc.Graph(figure={"data": fig,
                                               "layout": go.Layout(
                                                   autosize=True,
                                                   width=750,
                                                   height=450,
                                                   font={"family": "Raleway", "size": 10},
                                                   margin={"r": 0,
                                                           "t": 0,
                                                           "b": 45,
                                                           "l": 55},
                                                   showlegend=True,
                                                   titlefont={"family": "Raleway",
                                                              "size": 10},
                                                   xaxis={'title': 'Device ID',
                                                          'titlefont': {'family': 'Courier New, monospace',
                                                                        'size': 16,
                                                                        'color': 'black'}
                                                          },
                                                   yaxis={'title': 'No. Files Transmitted',
                                                          'titlefont': {'family': 'Courier New, monospace',
                                                                        'size': 16,
                                                                        'color': 'black'}
                                                          }),
                                               },
                                       config={"displayModeBar": False},
                                       style={"margin-top": "20px",
                                              "margin-bottom": "10px"}
                                       )
                             ], className="twelve columns",
                            style={"margin-left": "-57%",
                                   "margin-top": "20%"}
                            )


###############################################################################
# CALLBACK FOR LOCATION
###############################################################################
@app.callback(
    dash.dependencies.Output("inside-location", "children"),
    [Input('datatable-interactivity-location', 'data'),
     Input('datatable-interactivity-location', 'columns'),
     Input("datatable-interactivity-location", "derived_virtual_selected_rows")])
def update_figure_location(rows, columns, indices):
    """
    Callback for Location Plot
    """
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        print("location selected")
        selected = pred_df.values.tolist()
        if selected:
            fig = get_figure(selected)
            return html.Div([html.H6("Device Location Plot",
                                     className="subtitle padded"),
                             dcc.Graph(figure={"data": fig["data"],
                                               "layout":fig["layout"]},
                                       config={"displayModeBar": False})
                             ],
                            className="twelve columns",
                            style={"margin-left": "-57%",
                                   "margin-top": "20%"})


###############################################################################
# CALLBACK FOR BATTERY PERFORMANCE GRAPH
###############################################################################

@app.callback(
    dash.dependencies.Output("inside-battery", "children"),
    [Input('datatable-interactivity-battery', 'data'),
     Input('datatable-interactivity-battery', 'columns'),
     Input("datatable-interactivity-battery", "derived_virtual_selected_rows")])
def update_figure_battery(rows, columns, indices):
    """
    Callback for Battery Plot
    """
    fig = []
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        selected = pred_df.values.tolist()
        if selected:
            if len(selected) == 1:
                df_date = get_dataframe_for_plotting_transmission("data_downloaded/" + selected[0] + ".csv")
                device1 = plot_function(df_date)
                fig.append(device1)
            else:
                for each_fig in selected:
                    df_date = get_dataframe_for_plotting_transmission("data_downloaded/" + each_fig + ".csv")
                    device1 = plot_function(df_date)
                    fig.append(device1)
            return html.Div([html.H6("Device Battery Performance", className="subtitle padded"),
                             dcc.Graph(figure={"data": fig,
                                               "layout": go.Layout(
                                                   autosize=True,
                                                   width=750,
                                                   height=400,
                                                   font={"family": 'Courier New, monospace', "size": 10},
                                                   margin={"r": 30,
                                                           "t": 0,
                                                           "b": 35,
                                                           "l": 35},
                                                   showlegend=True,
                                                   titlefont={"family": "Raleway",
                                                              "size": 10},
                                                   xaxis={'title': 'Time ( Seconds )',
                                                          'titlefont': {'family': 'Courier New, monospace',
                                                                        'size': 16,
                                                                        'color': 'black'}},
                                                   yaxis={'title': 'Battery Level ( Percentage )',
                                                          'titlefont': {'family': 'Courier New, monospace',
                                                                        'size': 16,
                                                                        'color': 'black'}})},
                                       config={"displayModeBar": False},
                                       style={"margin-top": "20px",
                                              "margin-bottom": "10px"}
                                       ),
                             ], className="twelve columns",
                            style={"margin-left": "-59%",
                                   "margin-top": "30%"})


###############################################################################
# CALLBACK FOR DEVICE DETAILS
###############################################################################
@app.callback(
    dash.dependencies.Output("device-details-id", "children"),
    [Input('datatable-interactivity-device-details', 'data'),
     Input('datatable-interactivity-device-details', 'columns'),
     Input("datatable-interactivity-device-details", "derived_virtual_selected_rows")])
def update_figure_device_details(rows, columns, indices):
    """
    """
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        selected = pred_df.values.tolist()
        if selected:
            dataframe = pd.read_csv("data_downloaded/" + selected[0] + ".csv")
            df_details = pd.DataFrame()
            df_details["label"] = dataframe.columns
            df_details["value"] = dataframe.iloc[1, :].values.tolist()
            return [html.H6(["Device Details - " + selected[0]],
                            className="subtitle padded"),
                    html.Table(make_dash_table(df_details)), ]


###############################################################################
# CALLBACK FOR DOWNLOAD REPORT
###############################################################################


if __name__ == "__main__":
    app.run_server(debug=True)
