"""
Creating UI using Dash Library
"""
import subprocess
import os
import pickle
import base64
import datetime
import glob
from ftplib import FTP
import numpy as np
import dash
import dash.dependencies
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import generate_before_predict
import dash_table
import pymongo
from pymongo import MongoClient

#set the different colors
COLORS = {
    'background': '#111111',
    'text': '#7FDBFF'
}

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


################################################################################
                                #Main page#
################################################################################


app = dash.Dash()
app.config.suppress_callback_exceptions = True
IMAGE_FILENAME = 'pic.png' # replace with your own image
ENCODED_IMAGE_MAIN = base64.b64encode(open(IMAGE_FILENAME, 'rb').read())

app.layout = html.Div([

    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])
INDEX_PAGE = html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.H1('Wildly Listen', id='select',
                    style={'text-align': 'center',
                           'margin': '0 auto',})]),
        html.H2('Acoustic Monitoring and Audio Analysis')
    ]),
    html.Div(children=[
        html.Div(children=[
            html.Img(src='data:image/png;base64,{}'.format(ENCODED_IMAGE_MAIN),
                     style={'width':'100%',
                            'height':'250px'})
            ]),
        html.Div(id='t1', className="app-header", children=[
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
            html.Br(),
            html.Div(children=[
                dcc.Link('Input audio file', href='/page-1',
                         style={'fontSize': '100%',
                                'border':'3px solid green',
                                'color': 'white',
                                'text-decoration': 'none'}),
                html.Br()]),
            html.Br(),
            dcc.Link('FTP status', href='/page-3',
                     style={'fontSize': '100%',
                            'color': 'white',
                            'border':'3px solid green',
                            'position': 'relative',
                            'display':'inline',
                            'margin-top':100,
                            'text-decoration':'none'}),
            html.Br(),
            html.Br(),
            dcc.Link('Sound Library', href='/page-4',
                     style={'fontSize': '100%',
                            'color': 'white',
                            'border':'3px solid green',
                            'position': 'relative',
                            'text-decoration': 'none'}),
            html.Br()])]),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ',
                style={"position":"fixed",
                       "left":"0",
                       "bottom":"0",
                       "height":"2%",
                       "width":"100%",
                       "background-color":"black",
                       "color":"white",
                       "padding":"20px",
                       "textAlign":"center"})
    ])

################################################################################
                         #UPLOAD PAGE#
################################################################################


PAGE_1_LAYOUT = html.Div(id='out-upload-data', children=[
    html.Div(style={"background-color":"green",
                    "padding":"2px"}, children=[
                        dcc.Link('Home page', href='/', style={'fontSize': 20,
                                                               'color': 'white',
                                                               'text-decoration':'none'}),

                        html.H1('Upload Audio Files', style={'color': 'white',
                                                             'fontSize': 30,
                                                             'textAlign':'center',
                                                             'text-decoration':'none'})]),
    html.Div(id='display-play_1', children=[
        html.Br()]),
    dcc.Upload(
        id='upload-data1',
        children=
        html.Div(['Drag and Drop or',
                  html.A(' Select File')], style={'color': 'green',
                                                  'fontSize': 20,
                                                  'textAlign':'center'}),
        style={'color': 'green',
               'fontSize': 20,
               'border':'3px solid green',
               'textAlign':'center'},
        # Allow multiple files to be uploaded
        multiple=True
        ),
    html.Br(),
    dcc.Link('FTP status', href='/page-3',
             style={'text-decoration': 'none',
                    'textAlign': 'center',
                    'border':'3px solid green',
                    'margin-left':'800',
                    'align':'center',
                    'color': 'green',
                    'fontSize':20}),
    html.Div(id="page-1-content"),
    html.Div(id='page1', children=[
        html.Br(),
        html.Br()]),
    html.Footer('\xc2\xa9'+' Copyright WildlyTech Inc. 2019 .', style={"position":"fixed",
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
    with open('uploaded_files_from_dash/'+name, "wb") as file_p:
        file_p.write(base64.b64decode(data))


def parse_contents(contents, filename, date):
    """
    Read the file contents
    """
    # content_type, content_string = contents.split(',')
    if filename[-3:] == 'wav' or 'WAV':
        # create a director to store the uploaded files
        if not os.path.exists("uploaded_files_from_dash/"):
            os.makedirs("uploaded_files_from_dash/")
        # write the uploaded file into the directory created
        save_file(filename, contents)
        encoded_image_uploaded_file = 'uploaded_files_from_dash/'+ filename
        encoded_image_uploaded_file = base64.b64encode(open(encoded_image_uploaded_file, 'rb').read())
        embeddings = generate_before_predict.main('uploaded_files_from_dash/'+filename, 0, 0)
        predictions_prob, predictions = generate_before_predict.main('uploaded_files_from_dash/'+filename, 1, embeddings)
        predictions_prob = [float(i) for i in predictions_prob[0]]
        if predictions[0][0] == 1:
            output_sound = 'Motor sound'
        elif predictions[0][1] == 1:
            output_sound = 'Explosion sound '
        elif predictions[0][2] == 1:
            output_sound = 'Human sound'
        elif predictions[0][3] == 1:
            output_sound = 'Nature sound'
        elif predictions[0][4] == 1:
            output_sound = 'Domestic animal sound'
        elif predictions[0][5] == 1:
            output_sound = 'Tools sound'
        else:
            output_sound = 'None of the above'
        return  html.Div(style={'color': 'green', 'fontSize':14}, children=[
            html.Audio(id='myaudio',
                       src='data:audio/WAV;base64,{}'.format(encoded_image_uploaded_file),
                       controls=True,
                       title=True),
            html.H4('predictions rounded will be: '+ str(predictions[0])),
            html.H4('Predictions seems to be '+ output_sound,
                    style={'color':'green',
                           'fontSize': 30,
                           'textAlign':'center',
                           'text-decoration':'underline'}),
            dcc.Graph(id='example',
                      figure={
                          'data':[{'x':['Motor', 'Explosion', 'Human', 'Nature', 'Domestic', 'Tools'],
                                   'y':[i*100 for i in predictions_prob], 'marker':{
                                       'color':['rgba(26, 118, 255,0.8)', 'rgba(222,45,38,0.8)',
                                                'rgba(204,204,204,1)', 'rgba(0,150,0,0.8)',
                                                'rgba(204,204,204,1)', 'rgba(55, 128, 191, 0.7)']},
                                   'type':'bar'}],
                          'layout': {
                              'title':'probablistic prediction graph ',
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
                                  'title': 'Percenatge probabality',
                                  'titlefont':{
                                      'family':'Courier New, monospace',
                                      'size':18,
                                      'color':'green'}},
                              'height':400,
                              'paper_bgcolor':'rgba(0,0,0,0)',
                              'plot_bgcolor':'rgba(0,0,0,0)',
                              'font': {
                                  'color':'#7f7f7f'}}},
                      style={'marginBottom': 20,
                             'marginTop': 45,
                             'color':'black'}),
            html.P('Uploaded File : '+ filename,
                   style={'color': 'black',
                          'fontSize': 12})])
    else:
        return html.Div([
            html.Div(style={'color': 'blue', 'fontSize': 14}),
            html.H5('Unkown file type',
                    style={'marginBottom':20,
                           'marginTop':45,
                           'color': 'red',
                           'fontSize':14}),
            html.P('Uploaded File : '+filename, style={'color': 'black', 'fontSize': 12}),
            html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)),
                   style={'color':'black',
                          'fontSize': 12}),
        ])

# callback function for
@app.callback(Output(component_id='page-1-content', component_property='children'),
              [Input(component_id='upload-data1', component_property='contents')],
              [State('upload-data1', 'filename'),
               State('upload-data1', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    """
    check for upload of the files
    """
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


################################################################################
                            # TEST PAGE #
################################################################################


PAGE_2_LAYOUT = html.Div(id='Wildly listen', children=[
    html.Div(style={"background-color":"green", "padding":"2px"}, children=[
        html.H1('Acoustic Monitoring and Audio Analysis',
                style={'textAlign':'center',
                       'color':'white'})]),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Graph', 'value': 'graph'}],
        style={
            'width': '80%',
            'height': '60px'},
        value='select the task'),
    html.Div(id='page-2-content'),
    dcc.Link('Input audio file', href='/page-1',
             style={'marginBottom': 20,
                    'marginTop': 20,
                    'text-decoration':'none',
                    'fontSize': 14}),
    html.Br(),
    dcc.Link('Home Page', href='/',
             style={'marginBottom': 20,
                    'marginTop': 20,
                    'text-decoration':'none',
                    'fontSize':14}),
    html.Br(),
    dcc.Link('FTP status', href='/page-3',
             style={'marginBottom': 20,
                    'marginTop': 45,
                    'text-decoration':'none',
                    'fontSize': 14}),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ',
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
                             #FTP STATUS#
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
    ftp = FTP('34.211.117.196', user='******', passwd='*******')
    print "connected to FTP"
    ftp.cwd("Test_prototype/Nandi_hills/Nandi_hills_20_feb_2019/curve20_20_feb_2019/")
    ex = ftp.nlst()
    wav_files_only = check_for_wav_only(ex)
    dataframe = pd.DataFrame()
    dataframe["FileNames"] = wav_files_only
    dataframe = dataframe.sort_values(["FileNames"], ascending=[1])
    return dataframe

def call_for_data_1(dataframe):
    """
    Returns the data table consiting of all the wav files onto the webpage
    """
    return html.Div([
        html.H4("Total Number of Audio Clips : "+ str(dataframe.shape[0]),
                style={"color":"green",
                       'text-decoration':'underline'}),
        dash_table.DataTable(id='datatable-interactivity',
                             columns=[{"name": i,
                                       "id": i,
                                       "deletable": True} for i in dataframe.columns],
                             data=dataframe.to_dict("rows"),
                             filtering=True,
                             sorting=True,
                             # n_fixed_rows=1,
                             style_cell={'width': '50px'},
                             sorting_type="multi",
                             row_selectable="multi",
                             style_cell_conditional=[{'if': {'column_id': 'FileNames'},
                                                      'width': '50%'}],
                             style_table={"maxHeight":"400",
                                          "overflowY":"scroll"},
                             selected_rows=[]),
        html.Br(),
        html.Br()])


PAGE_3_LAYOUT = html.Div([
    html.Div(style={"background-color":"green", "padding":"2px"}, children=[
        html.H1("FTP Status", style={"color":"white",
                                     "text-align":"center",
                                     'fontSize': 20,
                                     'text-decoration':'underline'})]),
    html.Div(id='page-new-content'),
    html.Button("FTP Status", id='button'),
    html.Div(id="ftp_content_button"),
    html.Div(id="prediction-audio"),
    html.Div(id="datatable-interactivity-container")])


@app.callback(Output('page-new-content', 'children'),
              [Input('button', 'n_clicks')])
def ftp_data_display(n_clicks):
    """
    Wait for the click on the button
    """
    if n_clicks >= 1:
        dataframe = call_for_ftp()
        return call_for_data_1(dataframe)


@app.callback(Output('button', 'style'),
              [Input('button', 'n_clicks')])
def disabling_button(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}

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
        return html.Div([html.Button('Input Batch to Model', id='button_batch')])



def call_for_data_2(dataframe, list_of_malformed):
    """
    Returns the predicted values as the data table
    """
    return html.Div([
        html.H4("Total Number of Audio Clips : "+ str(dataframe.shape[0]),
                style={"color":"white",
                       'text-decoration':'underline'}),
        html.H4("Error while prediciton: " + str(list_of_malformed), style={"color":"white"}),
        dash_table.DataTable(id='datatable-interactivity-predictions',
                             columns=[{"name": i,
                                       "id": i,
                                       "deletable": True} for i in dataframe.columns],
                             data=dataframe.to_dict("rows"),
                             filtering=True,
                             sorting=True,
                             # n_fixed_rows=1,
                             style_cell={'width': '50px'},
                             sorting_type="multi",
                             row_selectable="single",
                             style_table={"maxHeight":"400",
                                          "overflowY":"scroll"},
                             selected_rows=[]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br()])

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
        encoded_image_to_play = base64.b64encode(open(path, 'rb').read())
        return html.Div([
            html.Br(),
            html.Audio(id='myaudio',
                       src='data:audio/WAV;base64,{}'.format(encoded_image_to_play),
                       controls=True,
                       title=True)])


@app.callback(
    Output('datatable-interactivity-container', 'children'),
    [Input('button_batch', 'n_clicks')])
def batch_downloading_and_predict(n_clicks):
    """
    Downloading the selected batch of files and processing them to model
    """
    if n_clicks >= 1:
        emb = []
        malformed = []
        dum_df = df.copy()
        for i in df["FileNames"].tolist():
            path = "FTP_downloaded/"+i
            if os.path.exists(path):
                print "path Exists"
            else:
                print "Downloading and generating embeddings ", i
                with open(path, 'wb') as file_obj:
                    ftp.retrbinary('RETR '+ i, file_obj.write)
            try:
                emb.append(generate_before_predict.main(path, 0, 0))
            except ValueError:
                print "malformed index", dum_df.loc[dum_df["FileNames"] == i].index
                dum_df = dum_df.drop(dum_df.loc[dum_df["FileNames"] == i].index)
                malformed.append(i)
                os.remove(path)
              # continue
        dum_df['features'] = emb
        if len(dum_df["FileNames"].tolist()) == 1:
            pred_prob, pred = generate_before_predict.main(path,
                                                           1,
                                                           np.array(dum_df.features.apply(
                                                               lambda x: x.flatten()).tolist()))
            dum_df["Motor_probability"] = "{0:.2f}".format(pred_prob[0][0])
            dum_df["Explosion_probability"] = "{0:.2f}".format(pred_prob[0][1])
            dum_df["Human_probability"] = "{0:.2f}".format(pred_prob[0][2])
            dum_df["Nature_probability"] = "{0:.2f}".format(pred_prob[0][3])
            dum_df["Domestic_probability"] = "{0:.2f}".format(pred_prob[0][4])
            dum_df["Tools_probability"] = "{0:.2f}".format(pred_prob[0][5])
            return call_for_data_2(dum_df[dum_df.drop("features",
                                                      axis=1).columns],
                                   malformed)

        elif len(dum_df["FileNames"] > 1):
            pred_prob, pred = generate_before_predict.main(path,
                                                           1,
                                                           np.array(dum_df.features.apply(
                                                               lambda x: x.flatten()).tolist()))
            dum_df["Motor_probability"] = ["{0:.2f}".format(i)
                                           for i in np.array(np.split(pred_prob,
                                                                      dum_df.shape[0]))[:, 0].tolist()]
            dum_df["Explosion_probability"] = ["{0:.2f}".format(i)
                                               for i in np.array(np.split(pred_prob,
                                                                          dum_df.shape[0]))[:, 1].tolist()]
            dum_df["Human_probability"] = ["{0:.2f}".format(i)
                                           for i in np.array(np.split(pred_prob,
                                                                      dum_df.shape[0]))[:, 2].tolist()]
            dum_df["Nature_probability"] = ["{0:.2f}".format(i)
                                            for i in np.array(np.split(pred,
                                                                       dum_df.shape[0]))[:, 3].tolist()]
            dum_df["Domestic_probability"] = ["{0:.2f}".format(i)
                                              for i in np.array(np.split(pred_prob,
                                                                         dum_df.shape[0]))[:, 4].tolist()]
            dum_df["Tools_probability"] = ["{0:.2f}".format(i)
                                           for i in np.array(np.split(pred_prob,
                                                                      dum_df.shape[0]))[:, 5].tolist()]
            return call_for_data_2(dum_df[dum_df.drop("features",
                                                      axis=1).columns],
                                   malformed)

        else:
            return html.Div([html.H3("Something went Wrong, Try again", style={"color":"white"}),
                             html.P("Note: If problem still persists file might be "+
                                    "corrupted or Input a valid 10 second .wav file",
                                    style={"color":"white"})])





####################################################################################
                                #Sound Library#
####################################################################################


EXPLOSION_SOUNDS = [
    'Fireworks',
    'Burst, pop',
    'Eruption',
    'Gunshot, gunfire',
    'Explosion',
    'Boom',
    'Fire'
]

MOTOR_SOUNDS = [
    'Chainsaw',
    'Medium engine (mid frequency)',
    'Light engine (high frequency)',
    'Heavy engine (low frequency)',
    'Engine starting',
    'Engine',
    'Motor vehicle (road)',
    'Vehicle'
]

WOOD_SOUNDS = [
    'Wood',
    'Chop',
    'Splinter',
    'Crack'
]

HUMAN_SOUNDS = [
    'Chatter',
    'Conversation',
    'Speech',
    'Narration, monologue',
    'Babbling',
    'Whispering',
    'Laughter',
    'Chatter',
    'Crowd',
    'Hubbub, speech noise, speech babble',
    'Children playing',
    'Whack, thwack',
    'Smash, crash',
    'Breaking',
    'Crushing',
    'Tearing',
    'Run',
    'Walk, footsteps',
    'Clapping'

]


DOMESTIC_SOUNDS = [
    'Dog',
    'Bark',
    'Howl',
    'Bow-wow',
    'Growling',
    'Bay',
    'Livestock, farm animals, working animals',
    'Yip',
    'Cattle, bovinae',
    'Moo',
    'Cowbell',
    'Goat',
    'Bleat',
    'Sheep',
    'Squawk',
    'Domestic animals, pets'

]


TOOLS_SOUNDS = [
    'Jackhammer',
    'Sawing',
    'Tools',
    'Hammer',
    'Filing (rasp)',
    'Sanding',
    'Power tool'
]


WILD_ANIMALS = [
    'Roaring cats (lions, tigers)',
    'Roar',
    'Bird',
    'Bird vocalization, bird call, bird song',
    'Squawk',
    'Pigeon, dove',
    'Chirp, tweet',
    'Coo',
    'Crow',
    'Caw',
    'Owl',
    'Hoot',
    'Gull, seagull',
    'Bird flight, flapping wings',
    'Canidae, dogs, wolves',
    'Rodents, rats, mice',
    'Mouse',
    'Chipmunk',
    'Patter',
    'Insect',
    'Cricket',
    'Mosquito',
    'Fly, housefly',
    'Buzz',
    'Bee, wasp, etc.',
    'Frog',
    'Croak',
    'Snake',
    'Rattle'
]

NATURE_SOUNDS = [
    "Silence",
    "Stream",
    "Wind noise (microphone)",
    "Wind",
    "Rustling leaves",
    "Howl",
    "Raindrop",
    "Rain on surface",
    "Rain",
    "Thunderstorm",
    "Thunder",
    'Crow',
    'Caw',
    'Bird',
    'Bird vocalization, bird call, bird song',
    'Chirp, tweet',
    'Owl',
    'Hoot'

]


TOTAL_LABELS = EXPLOSION_SOUNDS + NATURE_SOUNDS + MOTOR_SOUNDS + \
               HUMAN_SOUNDS + NATURE_SOUNDS  + DOMESTIC_SOUNDS + TOOLS_SOUNDS + WOOD_SOUNDS



CLIENT = MongoClient('localhost', 27017)
DATA_BASE = CLIENT.audio_library


PAGE_4_LAYOUT = html.Div([
    html.Div(style={"background-color":"green", "padding":"2px"}, children=[
        html.H1("Sound Library", style={"color":"white",
                                        "text-align":"center",
                                        'fontSize': 20,
                                        'text-decoration':'underline'})]),
    # html.Div(id='page-content'),
    html.H3('Select the Audio category', style={"color":"green",
                                                'text-decoration':'underline'}),
    dcc.Dropdown(id="select-class",
                 options=[
                     {"label":"Explosion", "value":"explosion_collection"},
                     {"label":"Motor Vehicle", "value":"motor_collection"},
                     {"label":"Human", "value":"human_collection"},
                     {"label":"Nature / Ambient", "value":"nature_collection"},
                     {"label":"Domestic animals", "value":"dom_collection"},
                     {"label":"Tools", "value":"tool_collection"},
                     {"label":"Wood", "value":"wood_collection"}],
                 multi=False,
                 placeholder="Search for Audio Category"),
    html.Div(id='output_dropdown'),
    html.Div(id='output_data'),
    html.Div(id="datatable-interactivity-container-2"),
    html.Div(id="page-4-content-2"),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ',
                style={"position":"fixed",
                       "left":"0",
                       "bottom":"0",
                       "height":"4%",
                       "width":"100%",
                       "background-color":"black",
                       "color":"white",
                       "padding":"20px",
                       "textAlign":"center"})])



def call_for_labels(data):
    """
    Returns the labels in format required by dropdown
    """
    labels = []
    for label in list(set(np.concatenate(data["labels_name"].values.tolist()))):
        if label in TOTAL_LABELS:
            labels.append({"label":str(label), "value":str(label)})
    return labels


@app.callback(Output('output_dropdown', 'children'),
              [Input('select-class', 'value')])
def select_class(value):
    """
    Return the sub-labels based on the selected class label
    """
    print value
    final_d = []
    global collection
    collection = DATA_BASE[str(value)]
    for labels in collection.find({}, {"labels_name":1}):
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



def call_for_data(dataframe):
    """
    Returns the data table based on the applied filters
    """
    return html.Div([
        html.P("Total Number of Audio Clips : "+ str(dataframe.shape[0]), style={"color":"green"}),
        dash_table.DataTable(id='datatable-interactivity',
                             columns=[{"name": i,
                                       "id": i,
                                       "deletable": True} for i in dataframe.columns],
                             data=dataframe.to_dict("rows"),
                             n_fixed_rows=1,
                             style_cell={'width':'50px'},
                             filtering=True,
                             sorting=True,
                             sorting_type="multi",
                             row_selectable="single",
                             selected_rows=[],
                             style_cell_conditional=[{'if': {'column_id': 'FileNames'},
                                                      'width': '75%'}])])




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



@app.callback(Output('output_data', 'children'),
              [Input('select-labels', 'value')])
def generate_layout(value):
    """
    Querying the results based on the applied selection
    """
    # print label_name
    print collection.count_documents({})
    print value
    global label_name
    label_name = value
    if value is None or len(value) == 0:
        return html.H5("Select any Label", style={"color":"green"})
    else:
        if len(value) == 1:
            final_d = []
            for each_name in collection.find({"labels_name":str(value[0])}):
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
            print type(data_frame['YTID'][0])
            return call_for_data(data_frame)
        elif len(value) == 2:
            final_d = []
            for each_name in collection.find({"$and":[{"labels_name":str(value[0])},
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
            return call_for_data(data_frame)
        else:
            return html.H5("Select Less number of Filter labels", style={"color":"green"})



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
    print "indices :", indices
    if indices is None:
        return html.Div(style={"padding":"20px"},
                        children=[html.P("Select Any audio ",
                                         style={"color":"green"})])
    else:
        global input_name
        path = subprocess.Popen("find /media/wildly/1TB-HDD/ -name "+\
                             df_inner.iloc[indices]["YTID"].astype(str)+"-"+\
                             df_inner.iloc[indices]["start_seconds"].astype(str)+"-"+\
                             df_inner.iloc[indices]["end_seconds"].astype(str)+".wav",
                                shell=True,
                                stdout=subprocess.PIPE)

        path = path.stdout.read().split("\n")[0]
        print "path ", path.split("\n")
        encoded_image_from_path = base64.b64encode(open(path, 'rb').read())
        print "len of indices ", len(indices)
        input_name = path
        return html.Div(style={"padding-bottom":"10%"}, children=[
            html.Br(),
            html.Br(),
            html.Audio(id='myaudio',
                       src='data:audio/WAV;base64,{}'.format(encoded_image_from_path),
                       controls=True,
                       title=True),
            html.Br(),
            html.Button('Input audio to model', id='button')])


# call back function for actual predictions and executing the model
@app.callback(Output('page-4-content-2', 'children'),
              [Input('button', 'n_clicks')])

def predict_on_downloaded_file(n_clicks):
    """
    actual predictions takes place here
    """
    print input_name
    if n_clicks >= 1:
        if input_name[-3:] == 'wav' or 'WAV':
            get_emb = generate_before_predict.main(input_name, 0, 0)
            predictions_prob, predictions = generate_before_predict.main(input_name,
                                                                         1,
                                                                         get_emb)
            predictions_prob = [float(i) for i in predictions_prob]
            if predictions[0] == 1:
                output_sound = 'Motor sound'
            elif predictions[1] == 1:
                output_sound = 'Explosion sound '
            elif predictions[2] == 1:
                output_sound = 'Human sound'
            elif predictions[3] == 1:
                output_sound = 'Nature sound'
            elif predictions[4] == 1:
                output_sound = 'Domestic animal sound'
            elif predictions[5] == 1:
                output_sound = 'Tools sound'
            else:
                output_sound = 'None of the above'
            return  html.Div(style={'color': 'green',
                                    'fontSize': 14,
                                    "padding-top":"-50%",
                                    "padding-bottom":"10px"},
                             children=[html.H4('predictions for: '+input_name.split("/")[-1]),
                                       html.H4('predictions rounded will be: '+ str(predictions)),
                                       html.H4('Predictions seems to be '+ output_sound,
                                               style={'color': 'green',
                                                      'fontSize': 30,
                                                      'textAlign': 'center',
                                                      'text-decoration':'underline'}),
                                       dcc.Graph(id='example',
                                                 figure={'data':[{'x':['Motor',
                                                                       'Explosion',
                                                                       'Human',
                                                                       'Nature',
                                                                       'Domestic',
                                                                       'Tools'],
                                                                  'y':[i*100 for i in predictions_prob],
                                                                  'marker':{'color':['rgba(26, 118, 255,0.8)', 'rgba(222,45,38,0.8)',
                                                                                     'rgba(204,204,204,1)', 'rgba(0,150,0,0.8)',
                                                                                     'rgba(204,204,204,1)', 'rgba(55, 128, 191, 0.7)']},
                                                                  'type':'bar'}],
                                                         'layout': {'title':'probablistic prediction graph ',
                                                                    'titlefont':{'family':'Courier New, monospace',
                                                                                 'size':22,
                                                                                 'color':'green'},

                                                                    'xaxis':{'title': 'Labels of the sound',
                                                                             'titlefont':{'family':'Courier New, monospace',
                                                                                          'size':18,
                                                                                          'color':'green'}},
                                                                    'yaxis':{'title':'Percenatge probabality',
                                                                             'titlefont':{'family':'Courier New, monospace',
                                                                                          'size':18,
                                                                                          'color':'green'}},
                                                                    'height':400,
                                                                    'paper_bgcolor':'rgba(0,0,0,0)',
                                                                    'plot_bgcolor':'rgba(0,0,0,0)',
                                                                    'font': {'color':'#7f7f7f'}}})])

# callback function for navigation settings
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
    else:
        return INDEX_PAGE

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
