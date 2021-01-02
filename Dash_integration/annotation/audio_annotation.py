"""
Audio Annotation Tool
"""
import argparse
import ast
import base64
import csv
import cv2
import dash
import dash.dependencies
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import glob
import librosa
import numpy as np
import os
import pandas as pd
import plotly.express
import plotly.graph_objs as go
import sys

from predictions.binary_relevance_model import generate_before_predict_BR,\
                                               get_results_binary_relevance,\
                                               predict_on_wavfile_binary_relevance


##########################################################################################
                       # INITIAL SETUP VARIABLES #
##########################################################################################
app = dash.Dash()
app.config.suppress_callback_exceptions = True
FILE_COUNT = 0
CONFIG_DATAS = {}
CSV_FILENAME = "New_annotation.csv"
CHECKLIST_DISPLAY = ["Bird", "Wind", "Vehicle", "Honking", "Conversation"]

# WHEN SUBMITTED FROM NEXT CONTENT
LABELS_LIST_CHECKLIST_NEXT = []
LABELS_LIST_DROPDOWN_NEXT = []


##########################################################################################
                                # INITIAL LAYOUT #
##########################################################################################

app.layout = html.Div([html.Div([html.H1("Audio Annotation",
                                         style={"color":"white", "display":"center", 'fontSize': 25, 'text-decoration':'underline'})],
                                style={"background-color":"green",
                                       "padding":"3px"}),
                       html.Div(dcc.Textarea(id="text_area",
                                             value="/home/" + os.getenv("USER") + "/Downloads",
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
                       html.Div(children=[html.Button("Start Annotation", id='button', n_clicks=0)],
                              style={"margin-left":"45%", "margin-top":"20px"}),
                       html.Br(),
                       html.Div(id="initial_submission"),
                       html.Div(id="next_submission"),
                       html.Div(id="prediction-audio"),
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
                                                                 n_clicks=0,
                                                                 style={"width" : "200px"})],
                                           style={"margin-left":'8%',
                                                  "width":"65%",
                                                  'display': 'inline-block'}),
                                        html.Div(children=[html.Button("Next Audio",
                                                                 value="next",
                                                                 id='next_button',
                                                                 n_clicks=0,
                                                                 style={"width" : "200px"})],
                                           style={"width":"10%", 'display': 'inline-block'})])




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

def model_prediction_tab():

    global NUMBER_OF_WAVFILES, FILE_COUNT, CONFIG_DATAS

    encoded_image_uploaded_file = NUMBER_OF_WAVFILES[FILE_COUNT]
    encoded_image_uploaded_file = base64.b64encode(open(encoded_image_uploaded_file, 'rb').read()).decode()
    embeddings = generate_before_predict_BR.main(NUMBER_OF_WAVFILES[FILE_COUNT], 0, 0, 0)

    ##############################################################################
            # Get label names
    ##############################################################################
    label_names = list(CONFIG_DATAS.keys())

    ##############################################################################
          # Implementing using the keras usual training techinque
    ##############################################################################

    prediction_probs, prediction_rounded = \
            predict_on_wavfile_binary_relevance.predict_on_embedding(\
                                                embedding = embeddings,
                                                label_names = label_names,
                                                config_datas = CONFIG_DATAS)

    output_sound = []
    for label_name, pred_round in zip(label_names, prediction_rounded):
        if pred_round == 1:
            output_sound += [label_name]

    if len(output_sound) == 1:
        output_sound = output_sound[0]
    elif len(output_sound) == 0:
        output_sound = 'None of the below'
    else:
        output_sound = str(output_sound)

    return  html.Div(style={'color': 'green', 'fontSize':14}, children=[
        html.Audio(id='myaudio',
                   src='data:audio/WAV;base64,{}'.format(encoded_image_uploaded_file),
                   controls=True,
                   style={"margin-top":"20px"}),
        html.H4('Predictions rounded will be: '+ str(prediction_rounded)),
        dcc.Graph(id='example',
                  figure={
                      'data':[{'x':label_names,
                               'y':prediction_probs, 'marker':{
                                   'color':'rgb(158,202,225)'},
                               'type':'bar',
                               "text":["{0:.2f}%".format(i) for i in prediction_probs],
                               "textposition" : 'auto',}],
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
                          'font': {
                              'color':'#7f7f7f'}}},
                  style={'marginBottom': 20,
                         'marginTop': 45,
                         'color':'black'}),
        html.P('Selected File : '+ NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
               style={'color': 'white',
                      'fontSize': 15})])

def spectrogram_tab():

    global NUMBER_OF_WAVFILES, FILE_COUNT

    clip, sr = librosa.load(NUMBER_OF_WAVFILES[FILE_COUNT])

    hop_length = int(0.01*sr)
    window_length = int(0.025*sr)
    n_fft = int(np.exp2(np.ceil(np.log2(window_length))))

    spec = librosa.feature.melspectrogram(clip, sr,
                                        n_mels=64,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        win_length=window_length)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    spec_db = cv2.resize(spec_db, (0, 0), fx=1, fy=4)

    fig = plotly.express.imshow(spec_db, origin='lower',
            title = 'Spectrogram: ' + \
                    NUMBER_OF_WAVFILES[FILE_COUNT].split('/')[-1],
            labels = {'x': 'Time (ms)',
                      'y': 'Frequency (Hz / {:.2f})'.format(sr/2/spec_db.shape[0]),
                      'color': 'Decibel'})

    return html.Div([dcc.Graph(figure=fig)],
                    style={"margin-top":"10%",
                           "text-align":"center"})

def annotation_tab(initial):
    try:
        global FILE_COUNT, LABELS_LIST_DROPDOWN_NEXT, NUMBER_OF_WAVFILES, LABELS_LIST_CHECKLIST_NEXT, LABELS_LIST_DROPDOWN_INITIAL
        LABELS_LIST_DROPDOWN_NEXT = []
        LABELS_LIST_CHECKLIST_NEXT = []
        LABELS_LIST_DROPDOWN_INITIAL = []

        if initial:
            FILE_COUNT = 0
            TOTAL_FOLDER_WAV_FILES = glob.glob(TEXT_PATH+"/*.wav")
            if os.path.exists(CSV_FILENAME):
                annotated_files = pd.read_csv(CSV_FILENAME, error_bad_lines=False)
                annotated_files = annotated_files['wav_file'].values.tolist()
                NUMBER_OF_WAVFILES = []
                for i in TOTAL_FOLDER_WAV_FILES:
                    if i.split("/")[-1] not in annotated_files:
                        # print(i)
                        NUMBER_OF_WAVFILES.append(i)
                print(len(NUMBER_OF_WAVFILES))
            else:
                NUMBER_OF_WAVFILES = TOTAL_FOLDER_WAV_FILES
            print("total wavfiles :", len(NUMBER_OF_WAVFILES))

        encoded_image_to_play = base64.b64encode(open(NUMBER_OF_WAVFILES[FILE_COUNT], 'rb').read()).decode()

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
                                              style={"margin-top":"20px",
                                                     "verticalAlign":"middle",
                                                     "margin-bottom":"30px"})]),
                         dash_table.DataTable(id='datatable-interactivity-' + ('inside' if initial else 'next'),
                                              columns=[{"name": i,
                                                        "id": i,
                                                        "deletable": True} for i in dataframe.columns],
                                              data=dataframe.to_dict("rows"),
                                              row_selectable="multi",
                                              style_table={"maxHeight":"300px",
                                                           "maxWidth" :"300px",
                                                           "overflowY":"scroll"},
                                              selected_rows=[]),
                         dcc.Dropdown(id="dropdown_data_" + ("initial" if initial else "next"),
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
                                             "margin-top": ("20" if initial else "40") + "px",
                                             "display":"inline-block",
                                             "font":"bold",
                                             "width":"40%"}),
                         dcc.Textarea(id="text_area_" + ("inside" if initial else "next"),
                                      placeholder='Selected Annotation',
                                      value="",
                                      style={"width":"50%",
                                             "margin-left":"25%",
                                             "fontSize":"16",
                                             "text-align":"center"}),
                         html.Div([html.Button("Submit",
                                               id="submit_" + ("initial" if initial else "next"),
                                               n_clicks=0,
                                               style={"width":"200px",
                                                      "margin-top":"10px"})],
                                  style={"text-align":"center"}),
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
    except ValueError:
        return html.Div([html.H5("Something wrong with Audio: -"+ NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                 style={"display":"center"})])


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
                       # Next: TABS CONTENT ON SELECTION #
##########################################################################################

@app.callback(Output('next_tab_content', 'children'),
              [Input("next-tabs-example", "value")])
def next_content_tab(value):
  """
  Displaying HTML page as per the TABS selection
  """
  if value == "annotation-tab":
      return annotation_tab(False)
  elif value == "spectrogram-tab":
      return spectrogram_tab()
  elif value == "model-prediction-tab":
      return model_prediction_tab()



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
      return annotation_tab(True)
  elif value == "spectrogram-tab":
      return spectrogram_tab()
  elif value == "model-prediction-tab":
      return model_prediction_tab()



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
        print(pred_df.values.tolist())
        LABELS_LIST_CHECKLIST_INITIAL = pred_df.values.tolist()
    else:
        pass
    if value_drop:
        LABELS_LIST_DROPDOWN_INITIAL.append(value_drop)
    else:
        pass
    return str(list(set(np.array(np.array(LABELS_LIST_CHECKLIST_INITIAL).flatten().tolist() + np.array(LABELS_LIST_DROPDOWN_INITIAL).flatten().tolist()).flatten())))



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
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Labels Name"]
        LABELS_LIST_CHECKLIST_NEXT = pred_df.values.tolist()
        print("Hello :", LABELS_LIST_CHECKLIST_NEXT)
    else:
        pass
    if value_drop:
        LABELS_LIST_DROPDOWN_NEXT.append(value_drop)
    else:
        pass
    return str(list(set(np.array(np.array(LABELS_LIST_CHECKLIST_NEXT).flatten().tolist() + np.array(LABELS_LIST_DROPDOWN_NEXT).flatten().tolist()).flatten())))


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
    all_filenames = annotated_files["wav_file"].values.tolist()
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
                    wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]] + ast.literal_eval(value))
                    print("submitted initial if")
                    print(value)
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
                wavfile_information_object.writerow(["wav_file",
                                                     "Label_1",
                                                     "Label_2",
                                                     "Label_3",
                                                     "Label_4"])
                wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]] + ast.literal_eval(value))
                print("submitted initial else")
                print(value)
                return html.Div([html.H5("Last Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                         style={"color":"white"})])



##########################################################################################
            # STATUS IN NEXT CONTENT
##########################################################################################
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
                    wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]] + ast.literal_eval(value))
                    print("submitted next if")
                    print(value)
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
                wavfile_information_object.writerow(["wav_file", "Label_1", "Label_2", "Label_3", "Label_4"])
                wavfile_information_object.writerow([NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1]]+ ast.literal_eval(value))
                print("submitted next else")
                print(value)
                return html.Div([html.H5("Last Submitted: - " +NUMBER_OF_WAVFILES[FILE_COUNT].split("/")[-1],
                                         style={"color":"white"})])




##########################################################################################
        # Main Function
##########################################################################################
if __name__ == '__main__':

    ##############################################################################
              # Description and Help
    ##############################################################################
    DESCRIPTION = 'Runs the Audio Annotation Tool.'

    ##############################################################################
              # Parsing the inputs given
    ##############################################################################
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()

    REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
    REQUIRED_NAMED.add_argument('-predictions_cfg_json',
                                '--predictions_cfg_json',
                                help='Input json configuration file for predictions output',
                                required=True)

    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    PARSED_ARGS = ARGUMENT_PARSER.parse_args()

    ##############################################################################
              # Import json data
    ##############################################################################
    CONFIG_DATAS = get_results_binary_relevance.import_predict_configuration_json(PARSED_ARGS.predictions_cfg_json)

    app.run_server(debug=True, use_reloader=True)
