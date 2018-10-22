# import necessary library functions
import dash
import dash.dependencies
from dash.dependencies import Input, Output,Event, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import pandas as pd
import random
from collections import deque
import pickle
import base64
import keras
import predict_on_wav_file
import io
import os
import ftp_test
import subprocess
import numpy as np
import datetime
from keras.models import Sequential, Model
import dash_table_experiments as dt
from keras.optimizers import RMSprop
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D, TimeDistributed, MaxPooling2D


#set the different colors
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Initialize the files and other necessary parameters
with open('file_count.pkl','wb') as f:
    pickle.dump(2,f)
with open('downloaded_from_ftp.pkl','wb') as f:
    pickle.dump([],f)
with open('list_of_files.pkl',  'wb') as f:
    pickle.dump(['.','..'],f)


################################################################################
                                #Main page
################################################################################


app = dash.Dash()
app.config.suppress_callback_exceptions = True
image_filename = '/home/shiv/Downloads/spec3.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content')
            ])

index_page = html.Div(style={'color': 'green', 'fontSize': 20},children=[
    html.H1('Wildly Listen',style={'color': 'green', 'fontSize': 50, 'textAlign':'center','text-decoration': 'underline'}),
    html.H2('A wildlife Conservation Tech company'),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image),  style={'width': '100%',
                'height': '450px'}),
    html.Br(),
    dcc.Link('predict on your audio file', href='/page-1',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'}),
    html.Br(),
    # dcc.Link('Explore more about model', href='/page-2',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'}),
    # html.Br(),
    dcc.Link('FTP status', href='/page-3',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'})
    ])



################################################################################
                         #page one Layout#
################################################################################



page_1_layout = html.Div(id='out-upload-data',children=[
        html.H1('Upload audio Files',style={'color': 'green', 'fontSize': 50,'textAlign': 'center'}),
        dcc.Upload(
        id='upload-data',
        children=
            html.Div(['Drag and Drop or',
            html.A(' Select Files',style={'color': 'green', 'fontSize': 20})]),
          style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
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
        dcc.Link('Explore More about model ', href='/page-2',style={'color': 'green', 'fontSize': 20,'textAlign': 'center'}),
        html.Br(),
        dcc.Link('Go Home', href='/',style={'color': 'green', 'fontSize': 20,'textAlign': 'center'}),
        html.Br(),
        dcc.Link('FTP status', href='/page-3',style={'marginBottom': 20, 'marginTop': 45,'color': 'green', 'fontSize': 20})
])




def parse_contents(contents,filename,date):
    content_type, content_string = contents.split(',')

    if filename[-3:] == 'wav':
        predictions_prob, predictions= predict_on_wav_file.main(filename)
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
        return  html.Div( style={'color': 'green', 'fontSize': 14},children=[
                # html.H4('predictions probabality :'+ str(predictions_prob)),
                html.H4('predictions rounded will be: '+ str(predictions)),
                html.H4('Predictions seems to be '+ output_sound,style={'color': 'green', 'fontSize': 30,'textAlign': 'center','text-decoration':'underline'}),
                dcc.Graph(id='example',
                             figure={
                                 'data':[{'x':['Motor','Explosion','Human','Nature','Domestic','Tools'],'y':[ i*100 for i in predictions_prob],    'marker':{
                                    'color':['rgba(26, 118, 255,0.8)', 'rgba(222,45,38,0.8)',
                                           'rgba(204,204,204,1)', 'rgba(0,150,0,0.8)',
                                           'rgba(204,204,204,1)','rgba(55, 128, 191, 0.7)']}    , 'type':'bar'}],
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
                             }},style={'marginBottom': 20, 'marginTop': 45, 'color':'black'}),
                html.P('Uploaded File : '+ filename,style={'color': 'black', 'fontSize': 12}),
                html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)),style={'color': 'black', 'fontSize': 12})
                # html.Button('Annotate the file',id='my-click',style={'color': 'black', 'fontSize': 12})
                ])
    else :
        return html.Div([
            html.Div( style={'color': 'blue', 'fontSize': 14}),
            html.H5('Unkown file type',style={'marginBottom': 20, 'marginTop': 45,'color': 'red', 'fontSize': 14}),
            html.P('Uploaded File : '+filename,style={'color': 'black', 'fontSize': 12}),
            html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)),style={'color': 'black', 'fontSize': 12}),
        ])


@app.callback(Output(component_id='page-1-content',component_property= 'children'),
              [Input(component_id='upload-data',component_property= 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



################################################################################
                            # page two layout #
################################################################################



page_2_layout = html.Div(id='Wildly listen',children=[
        html.H1('Its a wildlife Conservation Tech company',style={'textAlign': 'center','color':'green'}),
        dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Graph', 'value': 'graph'}
            # {'label': 'Try random Input for model', 'value': 'input for model'},
            # {'label': 'Mislabelled audio files', 'value' : 'mislabelled'}
            # {'label': 'Test Console ouput', 'value': 'model'}
            # {'label': 'About', 'value': 'about'}
        ],
        style={
        'width': '80%',
        'height': '60px'},
        value='select the task'
        ),
        html.Div(id = 'page-2-content'),
        dcc.Link('predict on your audio file', href='/page-1',style={'marginBottom': 20, 'marginTop': 20,'color': 'green', 'fontSize': 14}),
        html.Br(),
        dcc.Link('Go Home', href='/',style={'marginBottom': 20, 'marginTop': 20,'color': 'green', 'fontSize': 14}),
        html.Br(),
        dcc.Link('FTP status', href='/page-3',style={'marginBottom': 20, 'marginTop': 45,'color': 'green', 'fontSize': 14}),
    ])



@app.callback(Output('page-2-content', 'children'),
[Input('my-dropdown', 'value')])

def update_values(input_data):
    if input_data == 'graph':

        return     dcc.Graph(id='example',
                     figure={
                         'data':[{'x':[1,2,3,4],'y':[5,9,7,8],'type':'line'}],
                         'layout': {
                             'title':'Dash Plot',
                            'paper_bgcolor':'rgba(0,0,0,0)',
                            'plot_bgcolor':'rgba(0,0,0,0)',
                             'font': {
                                'color': colors['text']
                            }
                         }

                     },style={'marginBottom': 20, 'marginTop': 45})


###############################################################################
                             #page 3 layout#
###############################################################################



page_3_layout = html.Div(style={'color': 'green', 'fontSize': 20},children=[
    html.H1('FTP status',style={'color': 'green', 'fontSize': 50, 'textAlign':'center','text-decoration': 'underline'}),
    # html.Div(id = 'page-3-content-1'),
    html.Div(id = 'page-3-content-2'),
    dcc.Interval(
    id='interval-component',
    interval=1*15000, # in milliseconds
    n_intervals=0
    ),
    dcc.Link('predict on your audio file', href='/page-1',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'}),
    html.Br(),
    dcc.Link('Go Home', href='/',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'})

    ])



@app.callback(Output('page-3-content-2', 'children'),
              [Input('interval-component', 'n_intervals')])

def check_for_data(n_intervals):

    try :
        value, flag = ftp_test.ftp_data()
        with open('file_count.pkl','rb') as f:
            old_count = pickle.load(f)
        arb_count = old_count
        print 'Flag:',flag
        if flag == 1 :
            wav_file_count = ftp_test.download_files(value,flag)
            print 'Wav file count :', wav_file_count
            if wav_file_count >= 1:
                old_count = old_count + wav_file_count
                print 'New count :',old_count
                with open('file_count.pkl','wb') as f:
                    pickle.dump(old_count,f)
            if (arb_count + 1) == old_count:

                                return html.Div(children=[

                                       html.H2('New File uploaded..!!'),
                                       html.P('Downloading to local drive..',style={'marginBottom': 20, 'marginTop': 45,'color': 'black'})
                                       ])

            else :
                pass

        elif flag==0 and old_count >= 3 :

                            return html.Div(children=[

                                   html.H4('Total number of wav files downloaded :' + str(old_count - 2 ) ),
                                   dcc.Link('Run analysis', href='/page-4',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'})
                                   ])
        else :
            return html.Div(children = [
            html.H4('No wav Files are yet Uploaded',style={'marginBottom': 20, 'marginTop': 45,'color': 'black'})
            ])
    except :
        return html.Div(children = [
        html.H4('Error with FTP server: Directory timed out',style={'marginBottom': 20, 'marginTop': 45,'color': 'red'})
        ])


################################################################################
                            #page 4 Layout#
################################################################################

page_4_layout = html.Div(style={'color': 'green', 'fontSize': 20},children=[
    html.H2('Analysis on Downloaded files',style={'color': 'green', 'fontSize': 50, 'textAlign':'center','text-decoration': 'underline'}),
    html.Button('Play sound',id='button'),
    html.Div(id='display-play'),
    dcc.Interval(
    id='interval-component',
    interval=1*15000, # in milliseconds
    n_intervals=0
    ),
    dcc.Dropdown(
    id='my-dropdown'),
    html.Div(id = 'page-4-content-2'),
    dcc.Link('FTP status', href='/page-3',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'}),
    html.Br(),
    dcc.Link('predict on your audio file', href='/page-1',style={'marginBottom': 20, 'marginTop': 45,'color': 'green'})

    ])


# Make sound file play-able. 
@app.callback(Output('button', 'disabled'),
            [Input('my-dropdown','value')])

def button_enable_disable(dropdown_value):
    if dropdown_value :
        return False
    else :
        return True

@app.callback(Output('display-play','children'),
            [Input('button', 'n_clicks'), Input('my-dropdown','value')])

def play_sound(n_clicks,dropdown_value):
    print n_clicks
    if (dropdown_value != None) and   (n_clicks > 1) :

        from pydub import AudioSegment
        from pydub.playback import play

        song = AudioSegment.from_wav(dropdown_value)
        play(song)
        n_clicks = 0
        return


#Update the dropdown when-ever new file is downloaded
@app.callback(Output('my-dropdown', 'options'),
    [Input('interval-component', 'n_intervals')])

def markdown_drop_layout(n_intervals):
    with open('downloaded_from_ftp.pkl','rb') as f:
        filename = pickle.load(f)
        data_labels = []
    for each_file in   filename:
       data_labels.append({'label':each_file, 'value': each_file})

    return   data_labels


# Run the analysis on the selected wav file and plot the prediction graph
@app.callback(Output('page-4-content-2', 'children'),
[Input('my-dropdown', 'value')])

def predict_on_downloaded_file(input_name):
    print input_name

    if input_name[-3:] == 'wav':
        predictions_prob, predictions= predict_on_wav_file.main(input_name)
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
        return  html.Div( style={'color': 'green', 'fontSize': 14},children=[
                # html.H4('predictions probabality :'+ str(predictions_prob)),
                html.H4('predictions rounded will be: '+ str(predictions)),
                html.H4('Predictions seems to be '+ output_sound,style={'color': 'green', 'fontSize': 30,'textAlign': 'center','text-decoration':'underline'}),
                dcc.Graph(id='example',
                             figure={
                                 'data':[{'x':['Motor','Explosion','Human','Nature','Domestic','Tools'],'y':[ i*100 for i in predictions_prob],    'marker':{
                                    'color':['rgba(26, 118, 255,0.8)', 'rgba(222,45,38,0.8)',
                                           'rgba(204,204,204,1)', 'rgba(0,150,0,0.8)',
                                           'rgba(204,204,204,1)','rgba(55, 128, 191, 0.7)']}    , 'type':'bar'}],
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
                             }})
                             ])



# Navigation settings

@app.callback(dash.dependencies.Output('page-content', 'children'),
             [dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    if pathname=='/page-1':
        return page_1_layout
    elif pathname=='/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    elif pathname=='/page-4':
        return page_4_layout
    else :
        return index_page



if __name__=='__main__':
    app.run_server(debug=True,use_reloader=True)
