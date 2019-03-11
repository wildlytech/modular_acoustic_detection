"""
Creating UI using Dash Library
"""
import subprocess
import os
import pickle
import base64
import datetime
import dash
import glob
import dash.dependencies
from dash.dependencies import Input, Output,Event, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import predict_on_wav_file
import ftp_test
from ftplib import FTP
import dash_table
import pymongo
from pymongo import MongoClient
import numpy as np

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
    # with open('list_of_files.pkl', 'wb') as f:
    #   pickle.dump([], f)


################################################################################
                                #Main page
################################################################################


app = dash.Dash()
app.config.suppress_callback_exceptions = True
IMAGE_FILENAME = 'test_image.png' # replace with your own image
ENCODED_IMAGE = base64.b64encode(open(IMAGE_FILENAME, 'rb').read())

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
                     ])

INDEX_PAGE = html.Div(style={'color': 'green', 'fontSize': 20}, children=[
    html.Div(style={"background-color":"green", "padding":"2px"},children=[
    html.H1('Wildly Listen', style={'color': 'white', 'fontSize': 30, 'textAlign':'center', 'text-decoration': 'underline'}),
    html.H2('Acoustic Monitoring and Audio Analysis',style={'color': 'white', 'fontSize': 20,  'text-decoration': 'underline','textAlign':'center'})]),

    # html.Br(),
    html.Div(style={"background-color":"lavender", "padding-bottom":"100px"},children=[
    html.Div(style={ "background-color":"lightgrey"},children=[
    html.Img(src='data:image/png;base64,{}'.format(ENCODED_IMAGE), style={'width': '100%',
                                                                          'height':'800px' 
                                                                          })]),      
    dcc.Link('Input audio file', href='/page-1', style={'marginBottom': "20px", 'marginTop': "30px", 'color': 'green'}),
    html.Br(),
    dcc.Link('FTP status', href='/page-3', style={'marginBottom': 20, 'marginTop': 45, 'color':'green'}),
    html.Br(),
    dcc.Link('Sound Library', href='/page-4', style={'marginBottom':20, 'marginTop':45, 'color': 'green', 'fontSize':20})]),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ' ,style={"position":"fixed",
      "left":"0",
      "bottom":"0",
      "height":"4%",
      "width":"100%",
      "background-color":"black",
      "color":"white",
      "padding":"20px",
      "textAlign":"center"
      })   
    ])

################################################################################
                         #page one Layout#
################################################################################


PAGE_1_LAYOUT = html.Div(id='out-upload-data', children=[
    html.Div(style={"background-color":"green", "padding":"2px"},children=[
    html.H1('Upload audio Files', style={'color': 'white', 'fontSize': 30, 'textAlign': 'center'})]),
    html.Div(id='display-play_1'),
    dcc.Upload(
        id='upload-data',
        children=
        html.Div(['Drag and Drop or',
                  html.A(' Select File')], style={'color': 'green', 'fontSize': 20, 'textAlign':'center'}),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'texAlign': 'center',
            'margin': '10px',
            'color': 'green',
            'fontSize': 20,
            'font':'italic'
        },
        # Allow multiple files to be uploaded
        multiple=True
        ),
    html.Div(id='page-1-content'),
    html.Br(),
    dcc.Link('Home page', href='/', style={'color': 'green', 'fontSize': 20, 'textAlign':'center'}),
    html.Br(),
    dcc.Link('FTP status', href='/page-3', style={'marginBottom':20, 'marginTop':45, 'color': 'green', 'fontSize':20}),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 .' ,style={"position":"fixed",
      "left":"0",
      "bottom":"0",
      "height":"4%",
      "width":"100%",
      "background-color":"black",
      "color":"white",
      "padding":"20px",
      "textAlign":"center"
      })   

])


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open('uploaded_files_from_dash/'+name, "wb") as fp:
        fp.write(base64.b64decode(data))




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
        ENCODED_IMAGE = 'uploaded_files_from_dash/'+ filename
        ENCODED_IMAGE = base64.b64encode(open(ENCODED_IMAGE, 'rb').read())
        predictions_prob, predictions = predict_on_wav_file.main('uploaded_files_from_dash/'+filename)
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
        return  html.Div(style={'color': 'green', 'fontSize':14}, children=[
            html.Audio(id='myaudio', src='data:audio/WAV;base64,{}'.format(ENCODED_IMAGE), controls=True, title=True),  
            html.H4('predictions rounded will be: '+ str(predictions)),
            html.H4('Predictions seems to be '+ output_sound, style={'color': 'green', 'fontSize': 30, 'textAlign':'center', 'text-decoration':'underline'}),
            dcc.Graph(id='example',
                      figure={
                          'data':[{'x':['Motor', 'Explosion', 'Human', 'Nature', 'Domestic', 'Tools'], 'y':[i*100 for i in predictions_prob], 'marker':{
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
                                      'color':'green'}
                                       },
                              'yaxis':{
                                  'title': 'Percenatge probabality',
                                  'titlefont':{
                                      'family':'Courier New, monospace',
                                      'size':18,
                                      'color':'green'}
                                       },
                              'height':400,
                              'paper_bgcolor':'rgba(0,0,0,0)',
                              'plot_bgcolor':'rgba(0,0,0,0)',
                              'font': {
                                  'color':'#7f7f7f'
                                      }
                                      }
                                  },
                      style={'marginBottom': 20, 'marginTop': 45, 'color':'black'}),
            html.P('Uploaded File : '+ filename, style={'color': 'black', 'fontSize': 12}),
            # html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)), style={'color': 'black', 'fontSize': 12})
                        ])
    else:
        return html.Div([
            html.Div(style={'color': 'blue', 'fontSize': 14}),
            html.H5('Unkown file type', style={'marginBottom':20, 'marginTop':45, 'color': 'red', 'fontSize':14}),
            html.P('Uploaded File : '+filename, style={'color': 'black', 'fontSize': 12}),
            html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)), style={'color': 'black', 'fontSize': 12}),
        ])

# callback function for 
@app.callback(Output(component_id='page-1-content', component_property='children'),
              [Input(component_id='upload-data', component_property='contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
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
                            # page two layout #
################################################################################


PAGE_2_LAYOUT = html.Div(id='Wildly listen', children=[
    html.Div(style={"background-color":"green", "padding":"2px"},children=[
    html.H1('Acoustic Monitoring and Audio Analysis', style={'textAlign':'center', 'color':'white'})]),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Graph', 'value': 'graph'}
        ],
        style={
            'width': '80%',
            'height': '60px'},
        value='select the task'
        ),
    html.Div(id='page-2-content'),
    dcc.Link('Input audio file', href='/page-1', style={'marginBottom': 20, 'marginTop': 20, 'color':'green', 'fontSize': 14}),
    html.Br(),
    dcc.Link('Home Page', href='/', style={'marginBottom': 20, 'marginTop': 20, 'color': 'green', 'fontSize':14}),
    html.Br(),
    dcc.Link('FTP status', href='/page-3', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green', 'fontSize': 14}),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ' ,style={"position":"fixed",
      "left":"0",
      "bottom":"0",
      "height":"4%",
      "width":"100%",
      "background-color":"green",
      "color":"white",
      "padding":"20px",
      "textAlign":"center"
      })   
    ])

# callback function for dropdown with tet graph
@app.callback(Output('page-2-content', 'children'),
              [Input('my-dropdown', 'value')])
def update_values(input_data):
    """
    Test graph plot
    """
    if input_data == 'graph':

        return     dcc.Graph(id='example',
                             figure={
                                 'data':[{'x':[1, 2, 3, 4], 'y':[5, 9, 7, 8], 'type':'line'}],
                                 'layout': {
                                     'title':'Dash Plot',
                                     'paper_bgcolor':'rgba(0,0,0,0)',
                                     'plot_bgcolor':'rgba(0,0,0,0)',
                                     'font': {
                                         'color': COLORS['text']
                                    }}},
                             style={'marginBottom': 20, 'marginTop': 45})


###############################################################################
                             #page 3 layout#
###############################################################################


def call_for_ftp():
    global ftp
    ftp = FTP('****', user='*****', passwd='*****')
    print "connected to FTP"
    ex = ftp.nlst()


    dataframe = pd.DataFrame()
    dataframe["FileNames"] = ex
    dataframe = dataframe.sort_values(["FileNames"], ascending=[1])
    # dataframe["FileSize"] = EX
    return dataframe

def call_for_data(dataframe):
    return html.Div([
           html.H4("Total Number of Audio Clips : "+ str(dataframe.shape[0]), style={"color":"green",'text-decoration': 'underline'}),
           dash_table.DataTable(
                   id='datatable-interactivity',
                   columns=[
                          {"name": i, "id": i, "deletable": True} for i in dataframe.columns],
                      data=dataframe.to_dict("rows"),
                      filtering=True,
                      sorting=True,
                      n_fixed_rows=1,
                      style_cell={'width': '50px'},
                      sorting_type="multi",
                      row_selectable="single",
                      style_cell_conditional=[
                                  {'if': {'column_id': 'FileNames'},
                                   'width': '75%'}],
                                  # {'if': {'column_id': 'Region'},
                                  #  'width': '30%'}],
                      selected_rows=[]),
           html.Br(),
           html.Br(),
           dcc.Link('Home page', href='/', style={'color': 'green', 'fontSize': 20, 'textAlign':'center','text-decoration': 'underline'})])

# app = dash.Dash(__name__)
# app.config['suppress_callback_exceptions'] = True
PAGE_3_LAYOUT = html.Div([
    html.Div(style={"background-color":"green", "padding":"2px"},children=[
    html.H1("Sound Library", style={"color":"white", "text-align":"center",'fontSize': 20,'text-decoration': 'underline' })]),
    html.Div(id='page-new-content'),
    html.Button("FTP Status",id='button'),
    html.Div(id="datatable-interactivity-container"),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ' ,style={"position":"fixed",
      "left":"0",
      "bottom":"0",
      "height":"4%",
      "width":"100%",
      "background-color":"black",
      "color":"white",
      "padding":"20px",
      "textAlign":"center"
      })       ])


@app.callback(Output('page-new-content', 'children'),
              [Input('button', 'n_clicks')])

def ftp_data_display(n_clicks):
    if n_clicks>=1:
        dataframe = call_for_ftp()
        return call_for_data(dataframe)


@app.callback(Output('button', 'style'),
              [Input('button', 'n_clicks')])
def disabling_button(n_clicks):
  if n_clicks >=1:
    return {'display':"none"}



@app.callback(
    Output('datatable-interactivity-container', 'children'),
    [Input('datatable-interactivity', 'data'),
     Input('datatable-interactivity', 'columns'),
     Input("datatable-interactivity","derived_virtual_selected_rows")])
def display_output(rows, columns, indices):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    print "indices :", indices[0]
    path = "FTP_downloadeded/"+str(df.iloc[indices[0]]["FileNames"])
    print path
    if os.path.exists("FTP_downloadeded/"+df.iloc[indices[0]]["FileNames"]):
        print "path Exists"
    else:
        with open(path, 'wb') as file_obj:
            ftp.retrbinary('RETR '+ str(df.iloc[indices[0]]["FileNames"]), file_obj.write)

    # path = subprocess.call("find /media/wildly/1TB-HDD/ -name "+\
    #                      df.iloc[indices]["YTID"].astype(str)+"-"+df.iloc[indices]["start_seconds"].astype(str)+"-"+\
    #                      df.iloc[indices]["end_seconds"].astype(str)+".wav")

    ENCODED_IMAGE = base64.b64encode(open(path, 'rb').read())
    predictions_prob, predictions = predict_on_wav_file.main(path)
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
    return  html.Div(style={'color': 'green', 'fontSize':14}, children=[
        html.Br(),
        html.Audio(id='myaudio', src='data:audio/WAV;base64,{}'.format(ENCODED_IMAGE), controls=True, title=True),  
        html.H4('predictions rounded will be: '+ str(predictions)),
        html.H4('Predictions seems to be '+ output_sound, style={'color': 'green', 'fontSize': 30, 'textAlign':'center', 'text-decoration':'underline'}),
        dcc.Graph(id='example',
                  figure={
                      'data':[{'x':['Motor', 'Explosion', 'Human', 'Nature', 'Domestic', 'Tools'], 'y':[i*100 for i in predictions_prob], 'marker':{
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
                                  'color':'green'}
                                   },
                          'yaxis':{
                              'title': 'Percenatge probabality',
                              'titlefont':{
                                  'family':'Courier New, monospace',
                                  'size':18,
                                  'color':'green'}
                                   },
                          'height':400,
                          'paper_bgcolor':'rgba(0,0,0,0)',
                          'plot_bgcolor':'rgba(0,0,0,0)',
                          'font': {
                              'color':'#7f7f7f'
                                  }
                                  }
                              },
                  style={'marginBottom': 20, 'marginTop': 45, 'color':'black'})]) 
 
    print len(indices)
    if len(indices) == 0 :
        return None 
    else:
        return html.Div([
        html.Audio(id='myaudio', src='data:audio/WAV;base64,{}'.format(ENCODED_IMAGE), controls=True, title=True),
        html.Button('Input audio to model', id='button')])

################################################################################
                            #pag 4 Layou#
################################################################################


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


total_labels = EXPLOSION_SOUNDS + NATURE_SOUNDS + MOTOR_SOUNDS + HUMAN_SOUNDS + NATURE_SOUNDS  + DOMESTIC_SOUNDS + TOOLS_SOUNDS + WOOD_SOUNDS



client = MongoClient('localhost', 27017)
db = client.audio_library


PAGE_4_LAYOUT = html.Div([
    html.Div(style={"background-color":"green", "padding":"2px"},children=[
    html.H1("Sound Library", style={"color":"white", "text-align":"center",'fontSize': 20,'text-decoration': 'underline' })]),
    # html.Div(id='page-content'),
    html.H3('Select the Audio category', style={"color":"green",'text-decoration': 'underline' }),
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
                 placeholder="Search for Audio Category"
                 ),
    html.Div(id='output_dropdown'),
    html.Div(id='output_data'),
    html.Div(id="datatable-interactivity-container-2"),
    html.Div(id="page-4-content-2"),
    html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ' ,style={"position":"fixed",
      "left":"0",
      "bottom":"0",
      "height":"4%",
      "width":"100%",
      "background-color":"black",
      "color":"white",
      "padding":"20px",
      "textAlign":"center"
      })   
])



def call_for_labels(data):
    labels = []
    for label in list(set(np.concatenate(data["labels_name"].values.tolist()))):
        if label in total_labels:
            labels.append({"label":str(label), "value":str(label)})
    return labels


@app.callback(Output('output_dropdown', 'children'),
              [Input('select-class', 'value')])
def select_class(value):
    print value
    final_d = []
    global collection
    collection = db[str(value)]
    for labels in collection.find({}, {"labels_name":1}):
        final_d.append(labels)
    data = pd.DataFrame(final_d)
    # print data
    # time.sleep(2)
    labels = call_for_labels(data)
    return html.Div([
        html.P("Select the labels: Number of labels : " + str(len(labels)), style={"color":"green",'text-decoration': 'underline'}),
        dcc.Dropdown(id="select-labels",
                     options=labels,
                     multi=True,
                     # value=labels[0]["value"],
                     placeholder="Search for Label")])



def call_for_data(dataframe):
    return html.Div([
         html.P("Total Number of Audio Clips : "+ str(dataframe.shape[0]), style={"color":"green"}),
         dash_table.DataTable(
                   id='datatable-interactivity',
                   columns=[
                        {"name": i, "id": i, "deletable": True} for i in dataframe.columns],
                    data=dataframe.to_dict("rows"),
                      n_fixed_rows=1,
                      style_cell={'width':'50px'},
                    filtering=True,
                    sorting=True,
                    sorting_type="multi",
                    row_selectable="single",
                    selected_rows=[],
                    style_cell_conditional=[
                    {'if': {'column_id': 'FileNames'},
                     'width': '75%'}])])




@app.callback(Output('datatable-interactivity-container-2', 'style'),
              [Input('select-labels', 'value')])
def disabling_div_element(value):
  if value is None or len(value) == 0:
    return {"display":"none"}


@app.callback(Output('page-4-content-2', 'style'),
              [Input('select-labels', 'value'),Input("datatable-interactivity","derived_virtual_selected_rows")])
def disabling_div_element(value, indices):
  if value is None or len(value) == 0 or indices is None:
    return {"display":"none"}

@app.callback(Output('output_data', 'style'),
              [Input('select-labels', 'value')])
def disabling_div_element(value):
  if value is None or len(value) == 0:
    return {"display":"none"}





@app.callback(Output('output_data', 'children'),
              [Input('select-labels', 'value')])
def generate_layout(value):
    # print label_name
    print collection.count_documents({})
    print value
    global label_name 
    label_name = value
    if value is None or len(value) == 0:
        return html.H5("Select any Label",style={"color":"green"})
    else:
        if len(value) == 1:
            final_d = []
            for p in collection.find({"labels_name":str(value[0])}):
                final_d.append(p)
            try:
                data_frame = pd.DataFrame(final_d).drop(["positive_labels", "len"], axis=1)[["YTID","start_seconds","end_seconds", "labels_name"]].astype(str)
            except:
                data_frame = pd.DataFrame(final_d)[["YTID","start_seconds","end_seconds", "labels_name"]].astype(str)             
            print type(data_frame['YTID'][0])
            return call_for_data(data_frame)
        elif len(value) == 2:
            final_d = []
            for p in collection.find({"$and":[{"labels_name":str(value[0])}, {"labels_name":str(value[1])}]}):
                final_d.append(p)
            try:
                data_frame = pd.DataFrame(final_d).drop(["positive_labels", "len"], axis=1)[["YTID","start_seconds", "end_seconds", "labels_name"]].astype(str)
            except:
              data_frame = pd.DataFrame(final_d)[["YTID", "start_seconds", "end_seconds", "labels_name"]].astype(str)
            return call_for_data(data_frame)
        else:
            return html.H5("Select Less number of Filter labels",style={"color":"green"})



@app.callback(
    Output('datatable-interactivity-container-2', 'children'),
    [Input('datatable-interactivity', 'data'),
     Input('datatable-interactivity', 'columns'),
     Input("datatable-interactivity","derived_virtual_selected_rows")])
def display_output(rows, columns, indices):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    print "indices :", indices
    if indices is None:
        return html.Div(style={"padding":"20px"},children=[html.P("Select Any audio ",style={"color":"green"})])
    else:
        global input_name
        path = subprocess.Popen("find /media/wildly/1TB-HDD/ -name "+\
                             df.iloc[indices]["YTID"].astype(str)+"-"+df.iloc[indices]["start_seconds"].astype(str)+"-"+\
                             df.iloc[indices]["end_seconds"].astype(str)+".wav",shell=True, stdout=subprocess.PIPE)

        path = path.stdout.read().split("\n")[0]
        print "path ",path.split("\n")
        ENCODED_IMAGE = base64.b64encode(open(path, 'rb').read())
        print "len of indices ", len(indices)
        input_name = path
        return html.Div(style={"padding-bottom":"10%"}, children=[
          html.Br(),
          html.Br(),          
          html.Audio(id='myaudio', src='data:audio/WAV;base64,{}'.format(ENCODED_IMAGE), controls=True, title=True),
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
            predictions_prob, predictions = predict_on_wav_file.main(input_name)
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
            return  html.Div(style={'color': 'green', 'fontSize': 14,"padding-top":"-50%","padding-bottom":"10px"}, children=[
                    html.H4('predictions for: '+input_name.split("/")[-1]),
                    html.H4('predictions rounded will be: '+ str(predictions)),
                    html.H4('Predictions seems to be '+ output_sound,style={'color': 'green', 'fontSize': 30,'textAlign': 'center','text-decoration':'underline'}),
                    dcc.Graph(id='example',
                              figure={
                                     'data':[{'x':['Motor', 'Explosion', 'Human', 'Nature', 'Domestic', 'Tools'], 'y':[i*100 for i in predictions_prob], 'marker':{
                                         'color':['rgba(26, 118, 255,0.8)', 'rgba(222,45,38,0.8)',
                                                  'rgba(204,204,204,1)', 'rgba(0,150,0,0.8)',
                                                  'rgba(204,204,204,1)', 'rgba(55, 128, 191, 0.7)']}, 'type':'bar'}],
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
                                                 'color':'green'}
                                         },
                                         'yaxis':{
                                             'title': 'Percenatge probabality',
                                             'titlefont':{
                                                 'family':'Courier New, monospace',
                                                 'size':18,
                                                 'color':'green'}
                                         },
                                         'height':400,
                                         'paper_bgcolor':'rgba(0,0,0,0)',
                                         'plot_bgcolor':'rgba(0,0,0,0)',
                                         'font': {'color':'#7f7f7f'}
                                 }})
                                 ])

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


