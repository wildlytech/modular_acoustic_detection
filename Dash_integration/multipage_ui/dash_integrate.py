"""
Creating UI using Dash Library
"""
import subprocess
import os
import pickle
import base64
import datetime
import glob
import argparse
import numpy as np
from ftplib import FTP
import dash
import csv
import threading
import dash.dependencies
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table
from pymongo import MongoClient
import re

from predictions.binary_relevance_model import generate_before_predict_BR,\
                                               get_results_binary_relevance,\
                                               predict_on_wavfile_binary_relevance


########################################################################
                # set the different colors
########################################################################
COLORS = {
    'background': '#111111',
    'text': '#7FDBFF'
}



########################################################################
                # Description and Help
########################################################################
DESCRIPTION = "Creates UI for uploading Files and checking FTP server status. Works Offline and Online"
HELP = "Give the Required Arguments"



########################################################################
                  #parse the input arguments
########################################################################
ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()

REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
REQUIRED_NAMED.add_argument('-ftp_username', '--ftp_username', action='store',
                            help='Input FTP username', required=True)
REQUIRED_NAMED.add_argument('-ftp_password', '--ftp_password', action='store',
                            help='Input FTP Password', required=True)
REQUIRED_NAMED.add_argument('-remote_ftp_path', '--remote_ftp_path', action='store',
                            help='Directory path on remote FTP server', required=True)
REQUIRED_NAMED.add_argument('-local_folder_path', '--local_folder_path', action='store',
                            help='Directory path on local machine', required=True)

OPTIONAL_NAMED.add_argument('-ftp_host', '--ftp_host', action='store',
                            help='Input host name of FTP server', default='34.211.117.196')
OPTIONAL_NAMED.add_argument('-predictions_cfg_json',
                            '--predictions_cfg_json', action='store',
                            help='Input json configuration file for predictions output',
                            default='predictions/binary_relevance_model/binary_relevance_prediction_config.json')
OPTIONAL_NAMED.add_argument('-csv_filename', '--csv_filename', action='store',
                            help='Input the name of csv file to save results', default='wav_file_results.csv')

ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
PARSED_ARGS = ARGUMENT_PARSER.parse_args()



########################################################################
                      #set the FTP Taregt folder#
########################################################################
TARGET_FTP_FOLDER = PARSED_ARGS.remote_ftp_path
FOLDER_FILES_PATH = PARSED_ARGS.local_folder_path
CSV_FILENAME = PARSED_ARGS.csv_filename
FTP_USER_NAME = PARSED_ARGS.ftp_username
FTP_HOST_NAME = PARSED_ARGS.ftp_host
FTP_PASSWORD = PARSED_ARGS.ftp_password
PREDICTIONS_CFG_JSON = PARSED_ARGS.predictions_cfg_json

##############################################################################
          # Import json data
##############################################################################
CONFIG_DATAS = get_results_binary_relevance.import_predict_configuration_json(PREDICTIONS_CFG_JSON)


###############################################################################
                # Loop through all the models and get predictions
###############################################################################
def predictions_from_models(wavfile_path, embeddings):
    """
    Get predictions from embeddings
    """
    global CONFIG_DATAS

    prediction_probs, prediction_rounded = \
            predict_on_wavfile_binary_relevance.predict_on_embedding(\
                                                embedding = embeddings,
                                                label_names = list(CONFIG_DATAS.keys()),
                                                config_datas = CONFIG_DATAS)

    return prediction_probs, prediction_rounded

###############################################################################
        # Generates embeddings for each file and calls for predictions
###############################################################################
def get_predictions(wavfile_path):
    """
    Get predictions from wav file path
    """
    try:
        embeddings = generate_before_predict_BR.main(wavfile_path, 0, 0, 0)
    except:
        print(('\033[1m'+ "Predictions: " + '\033[0m' + "Error occured in File:- " + wavfile_path.split("/")[-1]))
        return None, None
    try:
        return predictions_from_models(wavfile_path, embeddings)
    except OSError:
        return None, None

def format_label_name(name):
    """
    Format string label name to remove negative label if it is
    EverythingElse
    """
    m = re.match("\[([A-Za-z0-9]+)\]Vs\[EverythingElse\]", name)

    if m is None:
        return name
    else:
        return m.group(1)

def get_formatted_detected_sounds(prediction_rounded):
    """
    Get names of detected output sounds as a single string.
    Array will be converted to comma-separated single string.
    No Elements will return None.
    """
    global CONFIG_DATAS

    # Determine which output sounds were detected
    output_sound = []
    for index, key in enumerate(CONFIG_DATAS.keys()):
        if prediction_rounded[index] == 1:
            output_sound += [key]

    # Format output sound variable to be string
    output_sound = [format_label_name(x) for x in output_sound]
    if len(output_sound) == 0:
        output_sound = 'None'
    else:
        output_sound = ', '.join(output_sound)

    return output_sound

def get_prediction_bar_graph(filepath):
    """
    Generate dash bar graph object based off predictions
    """
    global CONFIG_DATAS

    if not filepath.lower().endswith('.wav'):
        # Not a wav file so nothing to return
        return None

    prediction_probs, prediction_rounded = get_predictions(filepath)

    if prediction_probs is None:
        # Something went wrong with predictions, so exit
        return None

    output_sound = get_formatted_detected_sounds(prediction_rounded)

    return  prediction_probs, \
            prediction_rounded, \
            output_sound, \
            dcc.Graph(id='example',
                      figure={
                          'data':[{'x':[format_label_name(x) for x in list(CONFIG_DATAS.keys())],
                                   'y':["{0:.2f}".format(x) for x in prediction_probs],
                                   'text':["{0:.2f}%".format(x) for x in prediction_probs],
                                   'textposition':'auto',
                                   'marker':{
                                        'color':['rgba(26, 118, 255,0.8)', 'rgba(222,45,38,0.8)',
                                                 'rgba(204,204,204,1)', 'rgba(0,150,0,0.8)',
                                                 'rgba(204,204,204,1)', 'rgba(55, 128, 191, 0.7)']},
                                   'type':'bar'}],
                          'layout': {
                              'title':'probabilistic prediction graph ',
                              'titlefont':{
                                  'family':'Courier New, monospace',
                                  'size':22,
                                  'color':'green'},

                              'xaxis':{
                                  'title': 'Labels of the sound',
                                  'titlefont':{
                                      'family':'Courier New, monospace',
                                      'size':18,
                                      'color':'green'}},
                              'yaxis':{
                                  'title': 'Percentage probabality',
                                  'titlefont':{
                                      'family':'Courier New, monospace',
                                      'size':18,
                                      'color':'green'}},
                              'height':400,
                              'paper_bgcolor':'rgba(0,0,0,0)',
                              'plot_bgcolor':'rgba(0,0,0,0)',
                              'font': {'color':'#7f7f7f'}}},
                        style={'marginBottom': 20,
                               'marginTop': 45,
                               'color':'black'})


def get_data_table_from_dataframe(dataframe):
    """
    Returns the predicted values as the dash data table
    """

    # format numeric data into string format
    for column_name in dataframe.select_dtypes(include=[np.float]).columns:
        dataframe[column_name] = dataframe[column_name].apply(lambda x: "{0:.2f}%".format(x))

    return dash_table.DataTable(id='datatable-interactivity-predictions',
                                columns=[{"name": format_label_name(i),
                                          "id": i,
                                          "deletable": True} for i in dataframe.columns],
                                data=dataframe.to_dict("rows"),
                                style_header={"fontWeight": "bold"},
                                style_cell={'whiteSpace':'normal',
                                            'maxWidth': '240px'},
                                style_table={"maxHeight":"350px",
                                             "overflowY":"scroll",
                                             "overflowX":"auto"})

def check_pre_requiste_files():

    """
    check if wav files in FTP server are already
    downloaded locally
    """
    files_1 = glob.glob("FTP_downloaded/*.WAV")
    files_2 = glob.glob("FTP_downloaded/*.wav")
    files_1 = [i.split("/")[-1] for i in files_1]
    files_2 = [i.split("/")[-1] for i in files_2]
    files = files_1 + files_2
    with open('file_count.pkl', 'wb') as file_obj:
        pickle.dump(len(files), file_obj)
    with open('downloaded_from_ftp.pkl', 'wb') as file_obj:
        pickle.dump(files, file_obj)
    if not os.path.exists('uploaded_files_from_dash/'):
        os.makedirs('uploaded_files_from_dash/')



#############################################################################
                  # Sound Library Helper:
            # Returns the data table based on the applied filters
#############################################################################
def call_for_data(dataframe,
                  titleElementType,
                  titleColorStyle,
                  numPaddedLineBreaks=0,
                  list_of_malformed=None):
    """
    Returns the data table based on the applied filters
    """
    if list_of_malformed:
        list_of_malformed = ', '.join(list_of_malformed)
    else:
        list_of_malformed = "None"

    return html.Div([titleElementType("Total Number of Audio Clips : "+ str(dataframe.shape[0]),
                                      style={"color":titleColorStyle,
                                             "text-decoration":"underline"}),
                     titleElementType("Error while prediction: " + list_of_malformed,
                                      style={"color":"white"}),
                     get_data_table_from_dataframe(dataframe)] + \
                    [html.Br() for x in range(numPaddedLineBreaks)])



###############################################################################
                      #INDEX PAGE / HOME PAGE
###############################################################################


app = dash.Dash()
app.config.suppress_callback_exceptions = True
IMAGE_FILENAME = os.path.dirname(__file__)+'/pic.png' # replace with your own image
ENCODED_IMAGE_MAIN = base64.b64encode(open(IMAGE_FILENAME, 'rb').read()).decode()

app.layout = html.Div([dcc.Location(id='url',
                                    refresh=False),
                       html.Div(id='page-content')])

INDEX_PAGE = html.Div(children=[html.Div(children=[html.Div(children=[html.H1('Wildly Listen',
                                                                              id='select',
                                                                              style={'text-align': 'center',
                                                                                     'margin': '0 auto',})]),
                                                   html.H2('Acoustic Monitoring and Audio Analysis')]),
                                html.Div(children=[html.Div(children=[html.Img(src='data:image/png;base64,{}'.format(ENCODED_IMAGE_MAIN),
                                                                               style={'width':'100%',
                                                                                      'height':'250px'})]),
                                                   html.Div(id='t1',
                                                            className="app-header",
                                                            children=[html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      html.Div(children=[dcc.Link('Input audio file',
                                                                                                  href='/page-1',
                                                                                                  style={'fontSize': '100%',
                                                                                                         'color': 'solid green',
                                                                                                         'border':'3px solid green',
                                                                                                         'text-decoration': 'none'}),
                                                                                         html.Br()]),
                                                                      html.Br(),
                                                                      dcc.Link('FTP status',
                                                                               href='/page-3',
                                                                               style={'fontSize': '100%',
                                                                                      'color': 'solid green',
                                                                                      'border':'3px solid green',
                                                                                      'position': 'relative',
                                                                                      'display':'inline',
                                                                                      'margin-top':100,
                                                                                      'text-decoration':'none'}),
                                                                      html.Br(),
                                                                      html.Br(),
                                                                      dcc.Link('Sound Library',
                                                                               href='/page-4',
                                                                               style={'fontSize': '100%',
                                                                                      'color': 'solid green',
                                                                                      'border':'3px solid green',
                                                                                      'position': 'relative',
                                                                                      'text-decoration': 'none'}),
                                                                      html.Br()])]),
                                html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2020 ',
                                            style={"position":"fixed",
                                                   "left":"0",
                                                   "bottom":"0",
                                                   "height":"2%",
                                                   "width":"100%",
                                                   "background-color":"black",
                                                   "color":"white",
                                                   "padding":"20px",
                                                   "textAlign":"center"})])



###############################################################################
                         #1 UPLOAD PAGE
###############################################################################
PAGE_1_LAYOUT = html.Div(id='out-upload-data',
                         children=[html.Div(style={"background-color":"green",
                                                   "padding":"2px"},
                                            children=[dcc.Link('Home page',
                                                               href='/',
                                                               style={'fontSize': 20,
                                                                      'color': 'white',
                                                                      'text-decoration':'none'}),
                                                      html.H1('Upload Audio Files',
                                                              style={'color': 'white',
                                                                     'fontSize': 30,
                                                                     'textAlign':'center',
                                                                     'text-decoration':'none'})]),
                                   html.Div(id='display-play_1',
                                            children=[html.Br()]),
                                   dcc.Upload(id='upload-data1',
                                              children=html.Div(['Drag and Drop or',
                                                                 html.A(' Select File')],
                                                                style={'color': 'green',
                                                                       'fontSize': 20,
                                                                       'textAlign':'center'}),
                                              style={'color': 'green',
                                                     'fontSize': 20,
                                                     'border':'3px solid green',
                                                     'textAlign':'center'},
                                              # Allow multiple files to be uploaded
                                              multiple=True),
                                   html.Br(),
                                   html.Div(children=[html.Button('Folder Run',
                                                                  id='button1',
                                                                  n_clicks=0,
                                                                  style={'text-decoration': 'none',
                                                                         'textAlign': 'center',
                                                                         'border':'3px solid green',
                                                                         # 'margin-left':'800',
                                                                         # 'align':'center',
                                                                         'color': 'green',
                                                                         'fontSize':20}),
                                                      html.Br(),
                                                      html.Br()],
                                            style={"textAlign":"center"}),
                                   html.Div(children=[dcc.Link('FTP status',
                                                               href='/page-3',
                                                               style={'text-decoration': 'none',
                                                                      'textAlign': 'center',
                                                                      'border':'3px solid green',
                                                                      # 'margin-left':'800',
                                                                      # 'align':'center',
                                                                      'color': 'green',
                                                                      'fontSize':20})],
                                            style={"textAlign":"center"}),
                                   html.Div(id="page-1-content"),
                                   html.Div(id="page-1-content-link"),
                                   html.Div(id='page1',
                                            children=[html.Br(),
                                                      html.Br()]),
                                   html.Footer('\xc2\xa9'+' Copyright WildlyTech Inc. 2020 .',
                                               style={"position":"fixed",
                                                      "left":"0",
                                                      "bottom":"0",
                                                      "height":"4%",
                                                      "width":"100%",
                                                      "background-color":"black",
                                                      "color":"white",
                                                      "padding":"20px",
                                                      "textAlign":"center"})])


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open("uploaded_files_from_dash/"+name, "wb") as file_p:
        file_p.write(base64.b64decode(data))



def parse_contents(contents, filename, date):
    """
    Read the file contents
    """

    # content_type, content_string = contents.split(',')

    directory_path = "uploaded_files_from_dash/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    save_file(filename, contents)
    filepath = directory_path + filename
    encoded_image_uploaded_file = base64.b64encode(open(filepath, 'rb').read()).decode()

    bar_graph_info = get_prediction_bar_graph(filepath)

    if bar_graph_info is not None:
        # unpack bar graph information
        _, prediction_rounded, output_sound, bar_graph = bar_graph_info

        return  html.Div(style={'color': 'green', 'fontSize':14},
                         children=[ html.Audio(id='myaudio',
                                               src='data:audio/WAV;base64,{}'.format(encoded_image_uploaded_file),
                                               controls=True),
                                    html.H4('Predictions rounded will be: '+ str(prediction_rounded)),
                                    html.H4('Prediction seems to be '+ output_sound,
                                               style={'color':'green',
                                                      'fontSize': 30,
                                                      'textAlign':'center',
                                                      'text-decoration':'underline'}),
                                    bar_graph,
                                    html.P('Uploaded File : '+ filename)] + [html.Br() for x in range(3)])
    else:

        # Since the file is not a wav file or has problems, delete the file
        os.remove(filepath)

        return html.Div([
                          html.Div(style={'color': 'blue', 'fontSize': 14}),
                          html.H5('Unkown file type',
                                  style={'marginBottom':20,
                                         'marginTop':45,
                                         'color': 'red',
                                         'fontSize':14}),
                          html.P('Uploaded File : '+filename),
                          html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)),
                                 style={'color':'black',
                                        'fontSize': 12}),
                        ] + [html.Br() for x in range(3)])

def parse_contents_batch(contents, names, dates):
    """
    Multiple files that are uploaded are handled
    """
    global CONFIG_DATAS

    emb = []
    malformed = []
    dum_df = pd.DataFrame()
    dum_df['FileNames'] = names
    for i in zip(contents, names, dates):
        if not os.path.exists("uploaded_files_from_dash/"):
            os.makedirs("uploaded_files_from_dash/")
        path = "uploaded_files_from_dash/"+i[1]
        if os.path.exists(path):
            print("path Exists")
        else:
            print("Downloading and generating embeddings ", i[1])
            save_file(i[1], i[0])
            # with open(path, 'wb') as file_obj:
            #     ftp.retrbinary('RETR '+ i, file_obj.write)
        try:
            emb.append(generate_before_predict_BR.main(path, 0, 0, 0))
            os.remove(path)
        except ValueError:
            print("malformed index", dum_df.loc[dum_df["FileNames"] == i[1]].index)
            dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == i[1]].index)
            malformed.append(i[1])
            os.remove(path)
          # continue
    dum_df['features'] = emb
    if len(dum_df["FileNames"].tolist()) == 1:
        prediction_probs, prediction_rounded = predictions_from_models(path, np.array(dum_df.features.apply(lambda x: x.flatten()).tolist()))


        pred_df = pd.DataFrame(columns=["File Name"] + list(CONFIG_DATAS.keys()))
        pred_df.loc[0] = [dum_df["FileNames"].tolist()[0]]+ prediction_probs

        return call_for_data(pred_df,
                             titleElementType=html.H4,
                             titleColorStyle='white',
                             numPaddedLineBreaks=4,
                             list_of_malformed=malformed)

    elif len(dum_df["FileNames"] > 1):
        pred_df = pd.DataFrame(columns=["File Name"] + list(CONFIG_DATAS.keys()))

        for index, each_file, each_embedding in zip(list(range(0, dum_df.shape[0])), dum_df["FileNames"].tolist(), dum_df["features"].values.tolist()):
            try:
                prediction_probs, prediction_rounded = predictions_from_models(path, each_embedding)

                pred_df.loc[index] = [each_file] + prediction_probs
            except:
                pass

        return call_for_data(pred_df,
                             titleElementType=html.H4,
                             titleColorStyle='white',
                             numPaddedLineBreaks=4,
                             list_of_malformed=malformed)


# callback function for
@app.callback(Output(component_id='page-1-content', component_property='children'),
              [Input(component_id='upload-data1', component_property='contents')],
              [State('upload-data1', 'filename'),
               State('upload-data1', 'last_modified')])
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
            print("len of files: ", (list_of_names))
            return parse_contents_batch(list_of_contents, list_of_names, list_of_dates)


def start_batch_run_ftp_live(path_for_folder):
    """
    Writes the predicted results  on to csvfile row wise
    """
    global CONFIG_DATAS

    all_wav_files_path = glob.glob(path_for_folder+"*.WAV") + glob.glob(path_for_folder+"*.wav")
    all_wav_files = [each_file.split("/")[-1] for each_file in all_wav_files_path]
    print('len:', len(all_wav_files))
    dum_df = pd.DataFrame()
    dum_df["FileNames"] = all_wav_files
    tag_names = ["FileNames"] + list(CONFIG_DATAS.keys())


    # Check if the csv file is already existing or not. If it is existing then append the result
    # to same csv file based on the downloaded file
    csv_file_exists = os.path.exists(CSV_FILENAME)

    # Is there is no csv file then create one and write the result onto it.
    with open(CSV_FILENAME, "a" if csv_file_exists else "w") as file_object:
        wav_information_object = csv.writer(file_object)

        if not csv_file_exists:
            wav_information_object.writerow(tag_names)

        file_object.flush()

        # Loop over the files
        for each_file in dum_df['FileNames'].tolist():

            # Predict the result and save the result to the csv file
            pred_prob, pred = get_predictions(path_for_folder+each_file)

            wav_information_object.writerow([each_file] + ["{0:.2f}".format(x) for x in pred_prob])
            file_object.flush()

###############################################################################
                      #1A UPLOAD PAGE : Reloading for Folder file results#
###############################################################################

PAGE_5_LAYOUT = html.Div(html.Div([html.H4('Prediction Result'),
                                   html.Div(id='live-update-text'),
                                   dcc.Interval(id='interval-component',
                                                interval=1*1000,
                                                n_intervals=0)]))

@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def display_reloading_csv(n_intervals):
    """
    Reads csv file after every interval and displays the results
    """
    dataframe = pd.read_csv(CSV_FILENAME)

    return call_for_data(dataframe,
                         titleElementType=html.H4,
                         titleColorStyle='white',
                         numPaddedLineBreaks=4)

@app.callback(Output('button1', 'style'),
              [Input('button1', 'n_clicks')])
def disabling_button_1a(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}


@app.callback(Output('page-1-content', 'style'),
              [Input('button1', 'n_clicks')])
def disabling_button_1b(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}


@app.callback(Output('page-1-content-link', 'children'),
              [Input('button1', 'n_clicks')])
def disabling_button_1c(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        download_thread = threading.Thread(target=start_batch_run_ftp_live,
                                           args=[FOLDER_FILES_PATH])
        download_thread.start()
        return html.Div([html.Br(),
                         html.Br(),
                         dcc.Link("Selected Folder: "+FOLDER_FILES_PATH,
                                  href="/page-5",
                                  style={'text-decoration': 'none',
                                         'textAlign': 'left',
                                         'border':'3px solid green',
                                         # 'margin-left':'800',
                                         # 'align':'center',
                                         'color': 'green',
                                         'fontSize':20})],
                        style={'textAlign':"left"})


###############################################################################
                            # TEST PAGE / MISC#
###############################################################################


PAGE_2_LAYOUT = html.Div(id='Wildly listen', children=[
    html.Div(style={"background-color":"green", "padding":"2px"}, children=[
        html.H1('Acoustic Monitoring and Audio Analysis',
                style={'textAlign':'center',
                       'color':'white'})]),
    dcc.Dropdown(id='my-dropdown',
                 options=[{'label': 'Graph', 'value': 'graph'}],
                 style={'width': '80%',
                        'height': '60px'},
                 value='select the task'),
    html.Div(id='page-2-content'),
    dcc.Link('Input audio file',
             href='/page-1',
             style={'marginBottom': 20,
                    'marginTop': 20,
                    'text-decoration':'none',
                    'fontSize': 14}),
    html.Br(),
    dcc.Link('Home Page',
             href='/',
             style={'marginBottom': 20,
                    'marginTop': 20,
                    'text-decoration':'none',
                    'fontSize':14}),
    html.Br(),
    dcc.Link('FTP status',
             href='/page-3',
             style={'marginBottom': 20,
                    'marginTop': 45,
                    'text-decoration':'none',
                    'fontSize': 14}),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2020 ',
                style={"position":"fixed",
                       "left":"0",
                       "bottom":"0",
                       "height":"4%",
                       "width":"100%",
                       "background-color":"green",
                       "color":"white",
                       "padding":"20px",
                       "textAlign":"center"})])

# callback function for dropdown with tet graph
@app.callback(Output('page-2-content', 'children'),
              [Input('my-dropdown', 'value')])
def update_values(input_data):
    """
    Test graph plot
    """
    if input_data == 'graph':

        return dcc.Graph(id='example',
                         figure={'data':[{'x':[1, 2, 3, 4],
                                          'y':[5, 9, 7, 8],
                                          'type':'line'}],
                                 'layout': {'title':'Dash Plot',
                                            'paper_bgcolor':'rgba(0,0,0,0)',
                                            'plot_bgcolor':'rgba(0,0,0,0)',
                                            'font': {'color': COLORS['text']}}},
                         style={'marginBottom': 20,
                                'marginTop': 45})


###############################################################################
                   #FTP STATUS PAGE : Helper Functions
###############################################################################

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
    print("Connecting to FTP...")
    ftp = FTP(FTP_HOST_NAME, user=FTP_USER_NAME, passwd=FTP_PASSWORD)
    print("Connected to FTP!")
    ftp.cwd(TARGET_FTP_FOLDER)
    ex = ftp.nlst()
    wav_files_only = check_for_wav_only(ex)
    dataframe = pd.DataFrame()
    dataframe["FileNames"] = wav_files_only
    dataframe = dataframe.sort_values(["FileNames"], ascending=[1])
    return dataframe

###############################################################################
                   # FTP STATUS PAGE : Layout and callbacks
###############################################################################
PAGE_3_LAYOUT = html.Div([
    html.Div(style={"background-color":"green", "padding":"2px"},
             children=[html.H1("FTP Status",
                               style={"color":"white",
                                      "text-align":"center",
                                      'fontSize': 20,
                                      'text-decoration':'underline'})]),
    html.Div(id='page-new-content'),
    html.Button("FTP Status", id='button', n_clicks=0),
    html.Div(id="ftp_content_button"),
    html.Div(id="prediction-audio"),
    html.Div(id="datatable-interactivity-container")])



###############################################################################
                    # FTP STATUS PAGE: Callback:
            # Wait for button click & display FTP files
###############################################################################
@app.callback(Output('page-new-content', 'children'),
              [Input('button', 'n_clicks')])
def ftp_data_display(n_clicks):
    """
    Wait for the click on the button
    """
    if n_clicks >= 1:
        dataframe = call_for_ftp()
        return call_for_data(dataframe,
                             titleElementType=html.H4,
                             titleColorStyle='green',
                             numPaddedLineBreaks=2)


###############################################################################
                    # FTP STATUS PAGE: Callback:
         # Takes all the data that is been selected and stores as dataframe
###############################################################################
@app.callback(
    Output('ftp_content_button', 'children'),
    [Input('datatable-interactivity', 'data'),
     Input('datatable-interactivity', 'columns'),
     Input("datatable-interactivity", "derived_virtual_selected_rows")])
def display_output(rows, columns, indices):
    """
    Takes all the data that is been selected and stores as dataframe
    """
    global df
    if indices is not None and indices != []:
        df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        df = df.iloc[indices]
        return html.Div([html.Button('Input Batch to Model', id='button_batch', n_clicks=0)])



###############################################################################
                  # FTP STATUS PAGE : Callback:
            # Disabling div elements if none are selected
###############################################################################
@app.callback(
    Output('prediction-audio', 'style'),
    [Input('datatable-interactivity-predictions', 'data'),
     Input('datatable-interactivity-predictions', 'columns'),
     Input("datatable-interactivity-predictions", "derived_virtual_selected_rows")])
def play_button_for_prediction_disabling(rows, columns, indices):
    """
    Disabling play button if none are selected
    """
    if indices is None:
        return {"display":"none"}


@app.callback(Output('button', 'style'),
              [Input('button', 'n_clicks')])
def disabling_button(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}



###############################################################################
                  # FTP STATUS PAGE : Callback:
            # Playing the audio when file is selected
###############################################################################
@app.callback(
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
        path = "FTP_downloaded/"+str(pred_df.iloc[indices[0]]["FileNames"])
        encoded_image_to_play = base64.b64encode(open(path, 'rb').read()).decode()
        return html.Div([
            html.Br(),
            html.Audio(id='myaudio',
                       src='data:audio/WAV;base64,{}'.format(encoded_image_to_play),
                       controls=True)])



###############################################################################
                  # FTP STATUS PAGE : Callback:
      # Downloading the selected batch of files and processing them to model
###############################################################################
@app.callback(
    Output('datatable-interactivity-container', 'children'),
    [Input('button_batch', 'n_clicks')])
def batch_downloading_and_predict(n_clicks):
    """
    Downloading the selected batch of files and processing them to model
    """
    global CONFIG_DATAS

    if n_clicks >= 1:
        emb = []
        malformed = []
        dum_df = df.copy()
        for i in df["FileNames"].tolist():
            if not os.path.exists("FTP_downloaded/"):
                os.makedirs("FTP_downloaded/")
            path = "FTP_downloaded/"+i
            if os.path.exists(path):
                print("path Exists")
            else:
                print("Downloading and generating embeddings ", i)
                with open(path, 'wb') as file_obj:
                    ftp.retrbinary('RETR '+ i, file_obj.write)
            try:
                emb.append(generate_before_predict_BR.main(path, 0, 0, 0))
            except ValueError:
                print("malformed index", dum_df.loc[dum_df["FileNames"] == i].index)
                dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == i].index)
                malformed.append(i)
                os.remove(path)
              # continue
        dum_df['features'] = emb
        if len(dum_df["FileNames"].tolist()) == 1:
            prediction_probs, prediction_rounded = predictions_from_models(path, np.array(dum_df.features.apply(lambda x: x.flatten()).tolist()))


            pred_df = pd.DataFrame(columns=["File Name"] + list(CONFIG_DATAS.keys()))
            pred_df.loc[0] = [dum_df["FileNames"].tolist()[0]]+ prediction_probs

            return call_for_data(pred_df,
                                 titleElementType=html.H4,
                                 titleColorStyle='white',
                                 numPaddedLineBreaks=4,
                                 list_of_malformed=malformed)

        elif len(dum_df["FileNames"] > 1):
            pred_df = pd.DataFrame(columns=["File Name"] + list(CONFIG_DATAS.keys()))

            for index, each_file, each_embedding in zip(list(range(0, dum_df.shape[0])), dum_df["FileNames"].tolist(), dum_df["features"].values.tolist()):
                try:
                    prediction_probs, prediction_rounded = predictions_from_models(path, each_embedding)

                    pred_df.loc[index] = [each_file] + prediction_probs
                except:
                    pass

            return call_for_data(pred_df,
                                 titleElementType=html.H4,
                                 titleColorStyle='white',
                                 numPaddedLineBreaks=4,
                                 list_of_malformed=malformed)

        else:
            return html.Div([html.H3("Something went Wrong, Try again",
                                     style={"color":"white"}),
                             html.P("Note: If problem still persists file might be "+
                                    "corrupted or Input a valid 10 second .wav file",
                                    style={"color":"white"})])





###############################################################################
                    # SOUND LIBRARY - Requires MongoDB server
###############################################################################

CLIENT = MongoClient('localhost', 27017)
DATA_BASE = CLIENT.audio_library



###############################################################################
                 # Sound Library Page : Layout and callbacks
###############################################################################
PAGE_4_LAYOUT = html.Div([
    html.Div(style={"background-color":"green", "padding":"2px"},
             children=[html.H1("Sound Library",
                               style={"color":"white",
                                      "text-align":"center",
                                      'fontSize': 20,
                                      'text-decoration':'underline'})]),
    html.H3('Select the Audio category',
            style={"color":"green",
                   'text-decoration':'underline'}),
    dcc.Dropdown(id="select-class",
                 options=[{"label":i, "value":i} for i in list(CONFIG_DATAS.keys())],
                 multi=False,
                 placeholder="Search for Audio Category"),
    html.Div(id='output_dropdown'),
    html.Div(id='output_data'),
    html.Div(id="datatable-interactivity-container-2"),
    html.Div(id="page-4-content-2"),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2020 ',
                style={"position":"fixed",
                       "left":"0",
                       "bottom":"0",
                       "height":"4%",
                       "width":"100%",
                       "background-color":"black",
                       "color":"white",
                       "padding":"20px",
                       "textAlign":"center"})])




#############################################################################
  # Sound Library: Helper: Get the list of labels in dropdown format
#############################################################################
def call_for_labels(data):
    """
    Returns the labels in format required by dropdown
    """
    labels = []
    for label in list(set(np.concatenate(data["labels_name"].values.tolist()))):
        if label in list(CONFIG_DATAS.keys()):
            labels.append({"label":str(label), "value":str(label)})
    return labels



#############################################################################
                  # Sound Library: Callback:
          # Return the sub-labels based on the selected class label
#############################################################################
@app.callback(Output('output_dropdown', 'children'),
              [Input('select-class', 'value')])
def select_class(value):
    """
    Return the sub-labels based on the selected class label
    """
    print(value)
    final_d = []
    global COLLECTION
    COLLECTION = DATA_BASE[str(value)]
    for labels in COLLECTION.find({}, {"labels_name":1}):
        final_d.append(labels)
    data = pd.DataFrame(final_d)
    labels = call_for_labels(data)
    return html.Div([
        html.P("Select the labels: Number of labels : " + str(len(labels)),
               style={"color":"green",
                      'text-decoration':'underline'}),
        dcc.Dropdown(id="select-labels",
                     options=labels,
                     multi=True,
                     # value=labels[0]["value"],
                     placeholder="Search for Label")])



#############################################################################
                  # Sound Library: Callback:
            # Disables appropiate div element if none are selected
#############################################################################
@app.callback(Output('datatable-interactivity-container-2', 'style'),
              [Input('select-labels', 'value')])
def disabling_div_element(value):
    """
    Disables div element if none are selected
    """
    if value is None or len(value) == 0:
        return {"display":"none"}


@app.callback(Output('page-4-content-2', 'style'),
              [Input('select-labels', 'value'),
               Input("datatable-interactivity", "derived_virtual_selected_rows")])
def disabling_div_element_content_2(value, indices):
    """
    Disables div element based none selection
    """
    if value is None or len(value) == 0 or indices is None:
        return {"display":"none"}

@app.callback(Output('output_data', 'style'),
              [Input('select-labels', 'value')])
def disabling_div_output_data(value):
    """
    Disables the output div element based on none selection
    """
    if value is None or len(value) == 0:
        return {"display":"none"}



#############################################################################
                # Sound Library: Callback:
          # Querying MongoDB server based on selection
#############################################################################
@app.callback(Output('output_data', 'children'),
              [Input('select-labels', 'value')])
def generate_layout(value):
    """
    Querying the results based on the applied selection
    """
    # print label_name
    global label_name
    label_name = value
    if value is None or len(value) == 0:
        return html.H5("Select any Label", style={"color":"green"})
    else:
        if len(value) == 1:
            final_d = []
            for each_name in COLLECTION.find({"labels_name":str(value[0])}):
                final_d.append(each_name)
            try:
                data_frame = pd.DataFrame(final_d).drop(["positive_labels", "len"],
                                                        axis=1)[["YTID",
                                                                 "start_seconds",
                                                                 "end_seconds",
                                                                 "labels_name"]].astype(str)
            except:
                data_frame = pd.DataFrame(final_d)[["YTID",
                                                    "start_seconds",
                                                    "end_seconds",
                                                    "labels_name"]].astype(str)
            return call_for_data(data_frame, titleElementType=html.P, titleColorStyle='green')
        elif len(value) == 2:
            final_d = []
            for each_name in COLLECTION.find({"$and":[{"labels_name":str(value[0])},
                                                      {"labels_name":str(value[1])}]}):
                final_d.append(each_name)
            try:
                data_frame = pd.DataFrame(final_d).drop(["positive_labels", "len"],
                                                        axis=1)[["YTID",
                                                                 "start_seconds",
                                                                 "end_seconds",
                                                                 "labels_name"]].astype(str)
            except:
                data_frame = pd.DataFrame(final_d)[["YTID",
                                                    "start_seconds",
                                                    "end_seconds",
                                                    "labels_name"]].astype(str)
            return call_for_data(data_frame, titleElementType=html.P, titleColorStyle='green')
        else:
            return html.H5("Select Less number of Filter labels", style={"color":"green"})



#############################################################################
                # Sound Library Callback:
          # Displays the play button option and other option
#############################################################################
@app.callback(
    Output('datatable-interactivity-container-2', 'children'),
    [Input('datatable-interactivity', 'data'),
     Input('datatable-interactivity', 'columns'),
     Input("datatable-interactivity", "derived_virtual_selected_rows")])
def display_output_from_data(rows, columns, indices):
    """
    Displays the play button option and other option
    """
    df_inner = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is None:
        return html.Div(style={"padding":"20px"},
                        children=[html.P("Select Any audio ",
                                         style={"color":"green"})])
    else:
        global INPUT_NAME
        path = subprocess.Popen("find /media/wildly/1TB-HDD/ -name "+\
                             df_inner.iloc[indices]["YTID"].astype(str)+"-"+\
                             df_inner.iloc[indices]["start_seconds"].astype(str)+"-"+\
                             df_inner.iloc[indices]["end_seconds"].astype(str)+".wav",
                                shell=True,
                                stdout=subprocess.PIPE)

        path = path.stdout.read().split("\n")[0]
        print("path ", path.split("\n"))
        encoded_image_from_path = base64.b64encode(open(path, 'rb').read()).decode()
        print("len of indices ", len(indices))
        INPUT_NAME = path
        return html.Div(style={"padding-bottom":"10%"}, children=[
            html.Br(),
            html.Br(),
            html.Audio(id='myaudio',
                       src='data:audio/WAV;base64,{}'.format(encoded_image_from_path),
                       controls=True),
            html.Br(),
            html.Button('Input audio to model', id='button', n_clicks=0)])



#############################################################################
               # Sound Library: CallbacK:
          #actual predictions and executing the model
#############################################################################
@app.callback(Output('page-4-content-2', 'children'),
              [Input('button', 'n_clicks')])

def predict_on_downloaded_file(n_clicks):
    """
    actual predictions takes place here
    """
    global INPUT_NAME
    print(INPUT_NAME)

    if n_clicks >= 1:
        bar_graph_info = get_prediction_bar_graph(INPUT_NAME)

        if bar_graph_info is not None:

            # unpack bar graph information
            prediction_probs, prediction_rounded, output_sound, bar_graph = bar_graph_info

            filename = INPUT_NAME.split("/")[-1]

            return  html.Div(style={'color': 'green',
                                    'fontSize': 14,
                                    "padding-top":"-50%",
                                    "padding-bottom":"10px"},
                             children=[html.H4('predictions for: ' + filename),
                                       html.H4('Predictions rounded will be: '+ str(prediction_rounded)),
                                       html.H4('Prediction seems to be '+ output_sound,
                                               style={'color': 'green',
                                                      'fontSize': 30,
                                                      'textAlign': 'center',
                                                      'text-decoration':'underline'}),
                                       bar_graph,
                                       html.P('Uploaded File : '+ filename)] + [html.Br() for x in range(3)])



###############################################################################
               # callback function for navigation settings
###############################################################################

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    """
    Navigation setting
    """
    if pathname == '/page-1':
        return PAGE_1_LAYOUT
    elif pathname == '/page-2':
        return PAGE_2_LAYOUT
    elif pathname == '/page-3':
        return PAGE_3_LAYOUT
    elif pathname == '/page-4':
        return PAGE_4_LAYOUT
    elif pathname == "/page-5":
        return PAGE_5_LAYOUT
    else:
        return INDEX_PAGE



###############################################################################
                   #Main Function
###############################################################################

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
