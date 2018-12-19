"""
Creating UI using Dash Library
"""
import pickle
import base64
import datetime
import dash
import dash.dependencies
from dash.dependencies import Input, Output,Event, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import pandas as pd
import predict_on_wav_file
import ftp_test

#set the different colors
COLORS = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Initialize the files and other necessary parameters
with open('file_count.pkl', 'wb') as f:
    pickle.dump(0, f)
with open('downloaded_from_ftp.pkl', 'wb') as f:
    pickle.dump([], f)
with open('list_of_files.pkl', 'wb') as f:
    pickle.dump(['.', '..'], f)


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
    html.H1('Wildly Listen', style={'color': 'green', 'fontSize': 50, 'textAlign':'center', 'text-decoration': 'underline'}),
    html.H2('A Wildlife Conservation Technology solutions'),
    html.Img(src='data:image/png;base64,{}'.format(ENCODED_IMAGE), style={'width': '100%',
                                                                          'height':'450px'}),
    html.Br(),
    dcc.Link('Input audio file', href='/page-1', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green'}),
    html.Br(),
    dcc.Link('FTP status', href='/page-3', style={'marginBottom': 20, 'marginTop': 45, 'color':'green'})
    ])

################################################################################
                         #page one Layout#
################################################################################


PAGE_1_LAYOUT = html.Div(id='out-upload-data', children=[
    html.H1('Upload audio Files', style={'color': 'green', 'fontSize': 50, 'textAlign': 'center'}),
    html.Button('Play sound', id='button_1'),
    html.Div(id='display-play_1'),
    dcc.Upload(
        id='upload-data',
        children=
        html.Div(['Drag and Drop or',
                  html.A(' Select Files', style={'color': 'green', 'fontSize': 20})]),
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
    dcc.Link('FTP status', href='/page-3', style={'marginBottom':20, 'marginTop':45, 'color': 'green', 'fontSize':20})
])

def parse_contents(contents, filename, date):
    """
    Read the file contents
    """
    content_type, content_string = contents.split(',')

    if filename[-3:] == 'wav' or 'WAV':
        predictions_prob, predictions = predict_on_wav_file.main(filename)
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
            html.P('Last Modified : '+ str(datetime.datetime.fromtimestamp(date)), style={'color': 'black', 'fontSize': 12})
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

# callback function for play button disabling and enabling
@app.callback(Output('button_1', 'disabled'),
              [Input(component_id='upload-data', component_property='content')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def button_enable_disable(content, filename, date):
    """
    enabling and disabling the button
    """
    if filename[0]:
        return False
    else:
        return True

#callback function to play the audio
@app.callback(Output('display-play_1', 'children'),
              [Input('button_1', 'n_clicks'), Input(component_id='upload-data', component_property='content')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def play_sound(n_clicks, contents, filename,date):
    """
    audio playable option
    """
    print n_clicks
    print filename[0]
    if (filename[0] != None) and  (n_clicks > 0):
        from pydub import AudioSegment
        from pydub.playback import play

        song = AudioSegment.from_wav(filename[0])
        play(song)
        return

#call back function to check and update the clicks
@app.callback(Output('button_1', 'n_clicks'),
              [Input(component_id='upload-data', component_property='filename')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def play_sound(n_clicks, content, filename, date):
    """
    updating the clicks
    """
    if filename[0]:
        n_clicks = 0
        return n_clicks


################################################################################
                            # page two layout #
################################################################################


PAGE_2_LAYOUT = html.Div(id='Wildly listen', children=[
    html.H1('Its a wildlife Conservation Tech company', style={'textAlign':'center', 'color':'green'}),
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


PAGE_3_LAYOUT = html.Div(style={'color': 'green', 'fontSize': 20}, children=[
    html.H1('FTP status', style={'color': 'green', 'fontSize': 50, 'textAlign':'center', 'text-decoration':'underline'}),
    html.Div(id='page-3-content-2'),
    dcc.Interval(
        id='interval-component',
        interval=1*20000, # in milliseconds
        n_intervals=0
    ),
    dcc.Link('Input audio file', href='/page-1', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green'}),
    html.Br(),
    dcc.Link('Home Page', href='/', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green'})

    ])

# callback function for reloading page and scraping the remote ftp for .wav files
@app.callback(Output('page-3-content-2', 'children'),
              [Input('interval-component', 'n_intervals')])
def check_for_data(n_intervals):
    """
    check for .wav files in ftp server
    for every specified seconds
    """
    wav_files, wav_files_count, flag = ftp_test.ftp_data()
    with open('file_count.pkl', 'rb') as f:
        old_count = pickle.load(f)
    with open('downloaded_from_ftp.pkl', 'rb') as f:
        downloaded_files = pickle.load(f)
    if wav_files_count >= 1:
        downloaded_files = downloaded_files + wav_files
        with open('downloaded_from_ftp.pkl', 'w') as f:
            pickle.dump(downloaded_files, f)
    else:
        pass
    print 'Wav file count :', wav_files_count
    old_count = old_count + wav_files_count
    print 'New count :', old_count
    with open('file_count.pkl', 'wb') as f:
        pickle.dump(old_count, f)
    if wav_files_count == 1:
        return html.Div(children=[
            html.H2('New File uploaded..!!'),
            html.P('Downloading to local drive..', style={'marginBottom':20, 'marginTop':45, 'color':'black'})
                                 ])
    elif  wav_files_count == 0:
        return html.Div(children=[
            html.H4('Total number of wav files downloaded :' + str(old_count)),
            dcc.Link('Run analysis', href='/page-4', style={'marginBottom':20, 'marginTop':45, 'color':'green'})
                                  ])
    else:
        return  html.Div(children=[
            html.H4('Total number of wav files downloaded :' + str(old_count)),
            dcc.Link('Run analysis', href='/page-4', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green'})
                                  ])    

################################################################################
                            #pag 4 Layou#
################################################################################


PAGE_4_LAYOUT = html.Div(style={'color': 'green', 'fontSize': 20}, children=[
    html.H2('Analysis on Downloaded files', style={'color':'green', 'fontSize': 50, 'textAlign':'center', 'text-decoration':'underline'}),
    html.Button('Play sound', id='button'),
    html.Div(id='display-play'),
    dcc.Interval(
        id='interval-component',
        interval=1*15000, # in milliseconds
        n_intervals=0
    ),
    dcc.Dropdown(
        id='my-dropdown'),
    html.Div(id='page-4-content-2'),
    dcc.Link('FTP status', href='/page-3', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green'}),
    html.Br(),
    dcc.Link('Input audio file', href='/page-1', style={'marginBottom': 20, 'marginTop': 45, 'color': 'green'})

    ])

# calllback function for button diabling and enabling
@app.callback(Output('button', 'disabled'),
              [Input('my-dropdown', 'value')])
def button_enable_disable(dropdown_value):
    if dropdown_value:
        return False
    else:
        return True

# callback function for playing audio
@app.callback(Output('display-play', 'children'),
              [Input('button', 'n_clicks'), Input('my-dropdown', 'value')])
def play_sound(n_clicks, dropdown_value):
    """
    Playable option
    """
    print n_clicks
    if (dropdown_value != None) and (n_clicks > 0):
        from pydub import AudioSegment
        from pydub.playback import play
        song = AudioSegment.from_wav(dropdown_value)
        play(song)
        return

# callback for updating the clicks
@app.callback(Output('button', 'n_clicks'),
              [Input('my-dropdown', 'value')])
def play_sound(dropdown_value):
    """
    clicks modification
    """
    if dropdown_value:
        n_clicks = 0
        return n_clicks

# callback function for updating the dropdown with downloaded files
@app.callback(Output('my-dropdown', 'options'),
              [Input('interval-component', 'n_intervals')])
def markdown_drop_layout(n_intervals):
    """
    Adding downloaded wav files to markdown
    """
    with open('downloaded_from_ftp.pkl', 'rb') as file_obj:
        filename = pickle.load(file_obj)
        data_labels = []
    for each_file in   filename:
        data_labels.append({'label':each_file, 'value': each_file})

    return   data_labels


# call back function for actual predictions and executing the model
@app.callback(Output('page-4-content-2', 'children'),
              [Input('my-dropdown', 'value')])

def predict_on_downloaded_file(input_name):
    """
    actual predictions takes place here
    """
    print input_name
    if input_name[-3:] == 'wav':
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
        return  html.Div(style={'color': 'green', 'fontSize': 14}, children=[
                html.H4('predictions rounded will be: '+ str(predictions)),
                html.H4('Predictions seems to be '+ output_sound,style={'color': 'green', 'fontSize': 30,'textAlign': 'center','text-decoration':'underline'}),
                dcc.Graph(id='example',
                          figure={
                                 'data':[{'x':['Motor', 'Motor_2', 'Human', 'Nature', 'Domestic', 'Tools'], 'y':[i*100 for i in predictions_prob], 'marker':{
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

