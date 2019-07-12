"""
Audio Annotation Tool
"""
import os
import base64
import glob
import csv
from scipy.io import wavfile
import numpy as np
import dash
import dash.dependencies
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table
import matplotlib.pyplot as plt
import generate_before_predict


##########################################################################################
                       # INITIAL SETUP VARIABLES #
##########################################################################################
app = dash.Dash()
app.config.suppress_callback_exceptions = True
FILE_COUNT = 0
CSV_FILENAME = "New_annotation.csv"
CHECKLIST_DISPLAY = ["Nature", "Bird", "Wind", "Vehicle", "Honking", "Conversation"]


##########################################################################################
                                # INITIAL LAYOUT #
##########################################################################################

app.layout = html.Div([html.Div([html.H1("Audio Annotation",
                                         style={"color":"white",
                                     	   				"display":"center",
                                                'fontSize': 25,
                 										            'text-decoration':'underline'})],
                                style={"background-color":"green",
                                       "padding":"3px"}),
                       html.Div(dcc.Textarea(id="text_area",
                                             value="/home/shiv/Pictures",
                                             placeholder='Enter Path'),
                                style={"margin-top":"20px",
                                       "margin-left":"45%"}),
                       html.Div(dcc.Textarea(id="name_area",
                                             placeholder="User Name"),
                                style={"margin-top":"10px", "margin-left":"45%"}),
    		               html.Div(id='previous_next_button_display'),
    		               html.Div(id='initial_content_display'),
            		       html.Div(id='next_button_content_display'),
                       html.Div(id="intial_tab_content"),
                       html.Div(id="next_tab_content"),
            		       html.Div(id='previous_button_content_display'),
            		       html.Div(id="play_audio"),
                       html.Div(id="play_audio1"),
            		       html.Br(),
            		       html.Div(children=[html.Button("Start Annotation", id='button')],
                              style={"margin-left":"45%", "margin-top":"20px"}),
              		     html.Br(),
                       html.Div(id="initial_submission"),
                       html.Div(id="next_submission"),
                       html.Div(id="prediction-audio"),
                       html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ',
                                   style={"position":"fixed",
                                          "left":"0",
                                          "bottom":"0",
                                          "height":"2%",
                                          "width":"100%",
                                          "background-color":"black",
                                          "color":"white",
                                          "padding":"20px",
                                          "textAlign":"center"})])

##########################################################################################
                          # PREVIOUS and NEXT BUTTONS #
##########################################################################################

@app.callback(Output('previous_next_button_display', 'children'),
              [Input('button', 'n_clicks')])
def previous_next_button_content(n_clicks):
    """
    Previous and Next Audio click buttons
    """
    if n_clicks >= 1:        
        return html.Div(children=[html.Br(),
                                  html.Div(children=[html.Button("Previous Audio",
                                                                 value="previous",
                                                                 id='previous_button',
                                                                 style={"width" : "200px"})],
                                           style={"margin-left":'8%',
                                                  "width":"65%",
                                                  'display': 'inline-block'}),
        						              html.Div(children=[html.Button("Next Audio",
                                                                 value="next",
                                                                 id='next_button',
                                                                 style={"width" : "200px"})],
                                           style={"width":"10%",'display': 'inline-block'})])

##########################################################################################
                       # STYLING FOR TABS #
##########################################################################################

TABS_STYLES = {
    'height': '40px'
}
TAB_STYLE = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

TAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'green',
    'color': 'white',
    'padding': '6px'
}


##########################################################################################
                       # NEXT CONTENT TABS DISPLAY #
##########################################################################################

@app.callback(Output('next_button_content_display', 'children'),
              [Input('next_button', "n_clicks")])
def next_audio_content(n_clicks):	
    """
    Display the TABS for selection with default Value
    """
    if n_clicks >= 1:
        global FILE_COUNT, NUMBER_OF_WAVFILES
        FILE_COUNT = FILE_COUNT + 1
        return html.Div([dcc.Tabs(id="next-tabs-example",
                                  value='annotation-tab',
                                  children=[dcc.Tab(label='Spectrogram',
                                                    value='spectrogram-tab',
                                                    style=TAB_STYLE,
                                                    selected_style=TAB_SELECTED_STYLE),
                                            dcc.Tab(label='Annotation',
                                                    value='annotation-tab',
                                                    style=TAB_STYLE,
                                                    selected_style=TAB_SELECTED_STYLE),
                                            dcc.Tab(label="Model Predictions",
                                                    value="model-prediction-tab",
                                                    style=TAB_STYLE,
                                                    selected_style=TAB_SELECTED_STYLE)],
                                  style=TABS_STYLES)], style={"margin-top":"10px"})


##########################################################################################
                       # NEXT CONTENT TABS ON SELECTION #
##########################################################################################

@app.callback(Output('next_tab_content', 'children'),
              [Input("next-tabs-example", "value")])
def next_content_tab(value):
  """
  Displaying HTML page as per the TABS selection
  """
  if value == "annotation-tab":
        try:
            global FILE_COUNT, LABELS_LIST_DROPDOWN_NEXT, NUMBER_OF_WAVFILES, LABELS_LIST_CHECKLIST_NEXT, LABELS_LIST_DROPDOWN_INITIAL
            LABELS_LIST_DROPDOWN_NEXT = []
            LABELS_LIST_CHECKLIST_NEXT = []
            LABELS_LIST_DROPDOWN_INITIAL = []
            encoded_image_to_play = base64.b64encode(open(NUMBER_OF_WAVFILES[FILE_COUNT], 'rb').read())
            dataframe = pd.DataFrame()
            dataframe["Labels Name"] = CHECKLIST_DISPLAY
            return html.Div([html.Div([html.Br(),
                                       html.H2(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                               style={"text-align":"center",
                                                      "color":"green",
                                                      'text-decoration':'underline'}),
                                       html.Audio(id='myaudio',
                                                  src='data:audio/WAV;base64,{}'.format(encoded_image_to_play),
                                                  controls=True,
                                                  title=True,
                                                  style={"margin-top":"20px",
                                                         "verticalAlign":"middle",
                                                         "margin-bottom":"30px"})]),
                             dash_table.DataTable(id='datatable-interactivity-next',
                                                  columns=[{"name": i,
                                                            "id": i,
                                                            "deletable": True} for i in dataframe.columns],
                                                  data=dataframe.to_dict("rows"),
                                                  filtering=True,
                                                  sorting=True,
                                                  sorting_type="multi",
                                                  row_selectable="multi",
                                                  style_table={"maxHeight":"300px",
                                                               "maxWidth" :"300px",
                                                               "overflowY":"scroll"},
                                                  selected_rows=[]),
                             dcc.Dropdown(id="dropdown_data_next",
                                          options=[{'label': 'Nature ', 'value': 'Nature'},
                                                   {'label': 'Birds Chirping ', 'value': 'Bird'},
                                                   {'label': 'Wind Gushing', 'value': 'Wind'},
                                                   {'label': 'Vehicle  ', 'value': 'Vehicle'},
                                                   {'label': 'Honking  ', 'value': 'Honking'},
                                                   {'label': 'Conversation  ', 'value': 'Conversation'},
                                                   {'label': 'Dog Barking  ', 'value': 'Dog Barking'},
                                                   {'label': 'Tools  ', 'value': 'Tools'},
                                                   {'label': 'Axe  ', 'value': 'Axe'}],
                                          value="",
                                          placeholder="Search For Label..",
                                          style={"fontSize":"17",
                                                 "margin-top":"40px",
                                                 "display":"inline-block",
                                                 "font":"bold",
                                                 "width":"40%"}),
                             dcc.Textarea(id="text_area_next",
                                          placeholder='Selected Annotation',
                                          value="",
                                          style={"width":"50%",
                                                 "margin-left":"25%",
                                                 "fontSize":"16",
                                                 "text-align":"center"}),
                             html.Div([html.Button("Submit",
                                                   id="submit_next",
                                                   style={"width":"200px",
                                                          "margin-top":"10px"})],
                                      style={"text-align":"center"}),
                             html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ',
                                         style={"position":"fixed",
                                                "left":"0",
                                                "bottom":"0",
                                                "height":"2%",
                                                "width":"100%",
                                                "background-color":"black",
                                                "color":"white",
                                                "padding":"20px",
                                                "textAlign":"center"})])
        except ValueError:
            return html.Div([html.H5("Something wrong with Audio: -"+ NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                     style={"display":"center"})])

  elif value == "spectrogram-tab":
        sample_rate, samples = wavfile.read(NUMBER_OF_WAVFILES[FILE_COUNT])
        try:
            if samples.shape[1] == 2:
                samples = np.array([i[0] for i in samples])
        except:
            samples = samples
        plt.specgram(samples[:],
                     Fs=sample_rate,
                     xextent=(0, int(len(samples)/sample_rate)),
                     mode="psd",
                     cmap=plt.get_cmap('hsv'),
                     noverlap=5,
                     scale_by_freq=True)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1][:-4]+".jpg")
        encoded_image = base64.b64encode(open(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1][:-4]+".jpg", 'rb').read())
        os.remove(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1][:-4]+".jpg")

        return html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image))],
                        style={"margin-top":"10%",
                               "text-align":"center"})

  elif value == "model-prediction-tab":
        encoded_image_uploaded_file = NUMBER_OF_WAVFILES[FILE_COUNT]
        encoded_image_uploaded_file = base64.b64encode(open(encoded_image_uploaded_file, 'rb').read())
        embeddings = generate_before_predict.main(NUMBER_OF_WAVFILES[FILE_COUNT], 0, 0)
        predictions_prob, predictions = generate_before_predict.main(NUMBER_OF_WAVFILES[FILE_COUNT], 1, embeddings)
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
                       title=True,
                       style={"margin-top":"20px"}),
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
                                       'color':['black', 'rgb(158,202,225)',
                                                'rgb(158,202,225)', 'rgb(158,202,225)',
                                                'rgb(158,202,225)', 'rgb(158,202,225)']},
                                   'type':'bar',
                                   "text":["{0:.2f}".format(i*100) for i in predictions_prob],
                                   "textposition" : 'auto',}],
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
                              'font': {
                                  'color':'#7f7f7f'}}},
                      style={'marginBottom': 20,
                             'marginTop': 45,
                             'color':'black'}),
            html.P('Selected File : '+ NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                   style={'color': 'white',
                          'fontSize': 15})])



##########################################################################################
                       # INITIAL CONTENT TABS DISPLAY #
##########################################################################################

@app.callback(Output('initial_content_display', 'children'),
              [Input('text_area', 'value'),
               Input("button", "n_clicks")])
def initial_content(value, n_clicks):
    """
    Display the TABS for selection with default Value
    """
    if n_clicks >= 1:
        global TEXT_PATH
        TEXT_PATH = value
        return html.Div([dcc.Tabs(id="tabs-example",
                                  value='annotation-tab',
                                  children=[dcc.Tab(label='Spectrogram',
                                                    value='spectrogram-tab',
                                                    style=TAB_STYLE,
                                                    selected_style=TAB_SELECTED_STYLE),
                                            dcc.Tab(label='Annotation',
                                                    value='annotation-tab',
                                                    style=TAB_STYLE,
                                                    selected_style=TAB_SELECTED_STYLE),
                                            dcc.Tab(label="Model Predictions",
                                                    value="model-prediction-tab",
                                                    style=TAB_STYLE,
                                                    selected_style=TAB_SELECTED_STYLE)],
                                  style=TABS_STYLES)], style={"margin-top":"10px"})



##########################################################################################
                      # INITIAL CONTENT TABS SELECTION  #
##########################################################################################

@app.callback(Output('intial_tab_content', 'children'),
              [Input("tabs-example", "value")])
def initial_content_tab(value):
  """
  Returning HTML pages as per the TAB selection
  """
  if value == "annotation-tab":
        global FILE_COUNT, NUMBER_OF_WAVFILES, LABELS_LIST_DROPDOWN_INITIAL, TEXT_PATH, LABELS_LIST_CHECKLIST_NEXT, LABELS_LIST_DROPDOWN_NEXT
        LABELS_LIST_DROPDOWN_NEXT = []
        LABELS_LIST_CHECKLIST_NEXT = []
        LABELS_LIST_DROPDOWN_INITIAL = []
        FILE_COUNT = 0
        TOTAL_FOLDER_WAV_FILES = glob.glob(TEXT_PATH+"/*.wav")
        if os.path.exists(CSV_FILENAME):
            annotated_files = pd.read_csv(CSV_FILENAME, error_bad_lines=False)
            annotated_files = annotated_files['Filename'].values.tolist()
            NUMBER_OF_WAVFILES = []
            for i in TOTAL_FOLDER_WAV_FILES:
                if i.split("/")[-1] not in annotated_files:
                    # print(i)
                    NUMBER_OF_WAVFILES.append(i)
            print len(NUMBER_OF_WAVFILES)
        else:
            NUMBER_OF_WAVFILES = TOTAL_FOLDER_WAV_FILES
        print "total wavfiles :", len(NUMBER_OF_WAVFILES)
        encoded_image_to_play = base64.b64encode(open(NUMBER_OF_WAVFILES[FILE_COUNT], 'rb').read())
        dataframe = pd.DataFrame()
        dataframe["Labels Name"] = CHECKLIST_DISPLAY
        return html.Div([html.Div([html.Br(),
                                   html.H2(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                           style={"text-align":"center",
                                                  "color":"green",
                                                  'text-decoration':'underline'}),
                                   html.Audio(id='myaudio',
                                              src='data:audio/WAV;base64,{}'.format(encoded_image_to_play),
                                              controls=True,
                                              title=True,
                                              style={"margin-top":"20px",
                                                     "verticalAlign":"middle",
                                                     "margin-bottom":"30px"})]),
                         dash_table.DataTable(id='datatable-interactivity-inside',
                                              columns=[{"name": i,
                                                        "id": i,
                                                        "deletable": True} for i in dataframe.columns],
                                              data=dataframe.to_dict("rows"),
                                              filtering=True,
                                              sorting=True,
                                              sorting_type="multi",
                                              row_selectable="multi",
                                              style_table={"maxHeight":"300px",
                                                           "maxWidth" :"300px",
                                                           "overflowY":"scroll"},
                                              selected_rows=[]),
                         dcc.Dropdown(id="dropdown_data_initial",
                                      options=[{'label': 'Nature ', 'value': 'Nature'},
                                               {'label': 'Birds Chirping ', 'value': 'Bird'},
                                               {'label': 'Wind Gushing', 'value': 'Wind'},
                                               {'label': 'Vehicle  ', 'value': 'Vehicle'},
                                               {'label': 'Honking  ', 'value': 'Honking'},
                                               {'label': 'Conversation  ', 'value': 'Conversation'},
                                               {'label': 'Dog Barking  ', 'value': 'Dog Barking'},
                                               {'label': 'Tools  ', 'value': 'Tools'},
                                               {'label': 'Axe  ', 'value': 'Axe'}],
                                      value="",
                                      placeholder="Search For Label..",
                                      style={"fontSize":"17",
                                             "margin-top":"20px",
                                             "display":"inline-block",
                                             "font":"bold",
                                             "width":"40%"}),
                         dcc.Textarea(id="text_area_inside",
                                      placeholder='Selected Annotation',
                                      value="",
                                      style={"width":"50%",
                                             "margin-left":"25%",
                                             "fontSize":"16",
                                             "text-align":"center"}),
                         html.Div([html.Button("Submit",
                                               id="submit_initial",
                                               style={"width":"200px",
                                                      "margin-top":"10px"})],
                                  style={"text-align":"center"}),
                         html.Footer('\xc2\xa9'+ ' Copyright WildlyTech Inc. 2019 ',
                                     style={"position":"fixed",
                                            "left":"0",
                                            "bottom":"0",
                                            "height":"2%",
                                            "width":"100%",
                                            "background-color":"black",
                                            "color":"white",
                                            "padding":"20px",
                                            "textAlign":"center"})])

  elif value == "spectrogram-tab":
        sample_rate, samples = wavfile.read(NUMBER_OF_WAVFILES[FILE_COUNT])
        try:
            if samples.shape[1] == 2:
                samples = np.array([i[0] for i in samples])
        except:
            samples = samples
        plt.specgram(samples[:],
                     Fs=sample_rate,
                     xextent=(0, int(len(samples)/sample_rate)),
                     mode="magnitude",
                     cmap=plt.get_cmap('hsv'),
                     noverlap=5,
                     scale_by_freq=True)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1][:-4]+".jpg")
        encoded_image = base64.b64encode(open(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1][:-4]+".jpg", 'rb').read())
        os.remove(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1][:-4]+".jpg")

        return html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image))],
                        style={"margin-top":"10%",
                               "text-align":"center"})

  elif value == "model-prediction-tab":

        encoded_image_uploaded_file = NUMBER_OF_WAVFILES[FILE_COUNT]
        encoded_image_uploaded_file = base64.b64encode(open(encoded_image_uploaded_file, 'rb').read())
        embeddings = generate_before_predict.main(NUMBER_OF_WAVFILES[FILE_COUNT], 0, 0)
        predictions_prob, predictions = generate_before_predict.main(NUMBER_OF_WAVFILES[FILE_COUNT], 1, embeddings)
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
                       title=True,
                       style={"margin-top":"20px"}),
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
                                       'color':['black', 'rgb(158,202,225)',
                                                'rgb(158,202,225)', 'rgb(158,202,225)',
                                                'rgb(158,202,225)', 'rgb(158,202,225)']},
                                   'type':'bar',
                                   "text":["{0:.2f}".format(i*100) for i in predictions_prob],
                                   "textposition" : 'auto',}],
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
                              'font': {
                                  'color':'#7f7f7f'}}},
                      style={'marginBottom': 20,
                             'marginTop': 45,
                             'color':'black'}),
            html.P('Selected File : '+ NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                   style={'color': 'white',
                          'fontSize': 15})])


##########################################################################################
                     # SAVING ANNOTATIONS IN CSV FILE #
##########################################################################################


# WHEN SUBMITTED FROM INITIAL CONTENT
LABELS_LIST_CHECKLIST_INITIAL = []
LABELS_LIST_DROPDOWN_INITIAL = []
@app.callback(Output('text_area_inside', 'value'),
              [Input('datatable-interactivity-inside', 'data'),
               Input('datatable-interactivity-inside', 'columns'),
               Input("datatable-interactivity-inside", "derived_virtual_selected_rows"),
               Input("dropdown_data_initial", "value")])
def initial_content_annotation(rows, columns, indices, value_drop):
    """
    Displaying Selected Annotations on the text area : Initial Content
    """
    global LABELS_LIST_CHECKLIST_INITIAL, LABELS_LIST_DROPDOWN_INITIAL
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    LABELS_LIST_CHECKLIST_INITIAL = []
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Labels Name"]
        print pred_df.values.tolist()
    	LABELS_LIST_CHECKLIST_INITIAL = pred_df.values.tolist()
    else:
        pass
    if value_drop:
    	LABELS_LIST_DROPDOWN_INITIAL.append(value_drop)
    else:
    	pass
    return list(set(np.array(np.array(LABELS_LIST_CHECKLIST_INITIAL).flatten().tolist() + np.array(LABELS_LIST_DROPDOWN_INITIAL).flatten().tolist()).flatten()))


# WHEN SUBMITTED FROM NEXT CONTENT
LABELS_LIST_CHECKLIST_NEXT = []
LABELS_LIST_DROPDOWN_NEXT = []
@app.callback(Output('text_area_next', 'value'),
              [Input('datatable-interactivity-next', 'data'),
               Input('datatable-interactivity-next', 'columns'),
               Input("datatable-interactivity-next", "derived_virtual_selected_rows"),
               Input("dropdown_data_next", "value")])
def next_audio_content_annotation(rows, columns, indices, value_drop):
    """
    Displaying Selected Annotations on the text area : Next Content
    """
    global LABELS_LIST_CHECKLIST_NEXT, LABELS_LIST_DROPDOWN_NEXT
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    LABELS_LIST_CHECKLIST_INITIAL = []
    # print pred_df
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Labels Name"]
        LABELS_LIST_CHECKLIST_NEXT = pred_df.values.tolist()
    else:
        pass
    if value_drop:
        LABELS_LIST_DROPDOWN_NEXT.append(value_drop)
    else:
        pass
    return list(set(np.array(np.array(LABELS_LIST_CHECKLIST_NEXT).flatten().tolist() + np.array(LABELS_LIST_DROPDOWN_NEXT).flatten().tolist()).flatten()))


# WHEN SUBMITTED FROM PREVIOUS CONTENT
LABELS_LIST_CHECKLIST_PREVIOUS = []
LABELS_LIST_DROPDOWN_PREVIOUS = []
@app.callback(Output('text_area_previous', 'value'),
              [Input('checklist_data_previous', 'values'),
               Input("dropdown_data_previous", "value")])
def previous_audio_content_annotation(value,value_drop):
    """
    Displaying Selected Annotations on the text area : Previous Content
    """
    global LABELS_LIST_CHECKLIST_PREVIOUS, LABELS_LIST_DROPDOWN_PREVIOUS, CHECKLIST_DISPLAY
    LABELS_LIST_CHECKLIST_PREVIOUS = value
    if value_drop:
        CHECKLIST_DISPLAY.append(value_drop)
        LABELS_LIST_DROPDOWN_PREVIOUS.append(value_drop)
    else:
        pass
    return list(set(np.array(np.array(LABELS_LIST_CHECKLIST_PREVIOUS).flatten().tolist() + np.array(LABELS_LIST_DROPDOWN_PREVIOUS).flatten().tolist()).flatten()))


##########################################################################################
                  # DISABLING CONTENTS THAT ARE NOT REQUIRED #
##########################################################################################

@app.callback(Output('text_area', 'style'),
              [Input('button', 'n_clicks')])

def disabling_text_area(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}



@app.callback(Output('name_area', 'style'),
              [Input('button', 'n_clicks')])

def disabling_name_area(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}



@app.callback(Output('button', 'style'),
              [Input('button', 'n_clicks')])

def disabling_button(n_clicks):
    """
    Disabling the button after its being clicked once
    """
    if n_clicks >= 1:
        return {'display':"none"}


def check_for_duplicate(filename):
    """
    Checks if annotations are already submitted
    """
    annotated_files = pd.read_csv(CSV_FILENAME, error_bad_lines=False)
    all_filenames = annotated_files["Filename"].values.tolist()
    if filename in all_filenames:
        return True
    else:
        return False


@app.callback(Output('initial_submission', 'style'),
              [Input('submit_next', 'n_clicks')])
def disable_initial_submission(n_clicks):
    """
    Disbaling the content that is not required

    """
    if n_clicks >= 1:
        return {"display":"none"}


@app.callback(Output('initial_content_display', 'style'),
              [Input('next_button', 'n_clicks')])
def disable_initial_content(n_clicks):
    """
    Disbaling the content that is not required
    """
    global toggle_button_next
    if n_clicks >= 1:
        toggle_button_next = 1
        return {"display":"none"}



@app.callback(Output('intial_tab_content', 'style'),
              [Input('next_button', "n_clicks")])
def disable_initial_content_tab(n_clicks):
    '''
    Disbaling the content that is not required
    '''
    return {"display":"none"}



@app.callback(Output('previous_button_content_display', 'style'),
              [Input('next_button', 'value'),
               Input('previous_button', 'value')])
def disable_next_audio_content_enable_previous_audio_content(value_next, value_previous):
    """
    Disbaling the content that is not required
    """
    return {"display":"none"}

##########################################################################################
                           # SUBMISSION STATUS DISPLAY #
##########################################################################################

@app.callback(Output('initial_submission', 'children'),
              [Input('submit_initial', 'n_clicks'),
               Input("text_area_inside", "value")])

# STATUS IN INTIAL CONTENT
def submit_initial_button(n_clicks, value):
    """
    Disabling the button after its being clicked once
    """
    global FILE_COUNT
    if n_clicks >= 1:
        if os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, "a") as file_object:
                wavfile_information_object = csv.writer(file_object)
                if not check_for_duplicate(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]):
                    wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]] + value)
                    print "submitted initial if"
                    print value
                    return html.Div([html.H5("Last Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                             style={"color":"white",
                                                    "fontSize":"15"})])
                else:
      					    return html.Div([html.H5("Already Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                             style={"color":"white",
                                                    "fontSize":"15",
                                                    "margin-right":"70%"})])
        else:
            with open(CSV_FILENAME, "w") as file_object:
                wavfile_information_object = csv.writer(file_object)
                wavfile_information_object.writerow(["Filename",
                                                     "Label1",
                                                     "Label2",
                                                     "Label3",
                                                     "Label4"])
                wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]] + value)
                print "submitted initial else"
                print value
                return html.Div([html.H5("Last Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                         style={"color":"white"})])



# STATUS IN NEXT CONTENT
@app.callback(Output('next_submission', 'children'),
              [Input('submit_next', 'n_clicks'),
               Input("text_area_next", "value")])
def submit_next_button(n_clicks, value):
    """
    Disabling the button after its being clicked once
    """
    global FILE_COUNT, LABELS_LIST_CHECKLIST_NEXT, LABELS_LIST_DROPDOWN_NEXT
    if n_clicks >= 1:
        if os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, "a") as file_object:
                wavfile_information_object = csv.writer(file_object)
                if not check_for_duplicate(NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]):
                    wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]] + value)
                    print "submitted next if"
                    print value
                    return html.Div([html.H5("Last Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                             style={"color":"white",
                                                    "fontSize":"15"})])
                else:
          			    return html.Div([html.H5("Already Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                             style={"color":"white",
                                                    "fontSize":"15"})])
        else:
            with open(CSV_FILENAME, "w") as file_object:
                wavfile_information_object = csv.writer(file_object)
                wavfile_information_object.writerow(["Filename", "Label1", "Label2", "Label3", "Label4"])
                wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]]+value)
                print "submitted next else"
                print value
                return html.Div([html.H5("Last Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                         style={"color":"white"})])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
