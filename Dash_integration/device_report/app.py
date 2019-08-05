# -*- coding: utf-8 -*-
import glob
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd 
import numpy as np
import dash.dependencies
from plotly import tools
import plotly.graph_objs as go
from pages import overview
from utils import Header, make_dash_table
import pathlib
import dash_table
import argparse

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()


##########################################################################################
                         # READING ARGUMENTS #
##########################################################################################

DESCRIPTION = "Wildly Acoutsic Monitoring Device Report"
HELP = "Give the Required Arguments"

#parse the input arguments given from command line
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-transmission_mode_csv_path', '--transmission_mode_csv_path', action='store',
                    help=HELP,default="transmission_details")
PARSER.add_argument('-record_mode_csv_path', '--record_mode_csv_path', action='store',
                    help='Input the path')
RESULT = PARSER.parse_args()




##########################################################################################
                         # READING DATAFRAMES #
##########################################################################################

DF_HIST_PRICES = pd.read_csv(DATA_PATH.joinpath("df_hist_prices.csv"))
DF_FUND_FACTS = pd.read_csv(DATA_PATH.joinpath("df_fund_facts.csv"))

if RESULT.transmission_mode_csv_path:
    csv_files = glob.glob(RESULT.transmission_mode_csv_path+"/*.csv")
else:
    pass
if RESULT.record_mode_csv_path:
    csv_files1 = glob.glob(RESULT.record_mode_csv_path+"/*.csv")
else:
    pass

def get_dictionary(csv_files):
    req_dict = dict()
    for index, each_file in enumerate(csv_files):    
        req_dict["Device_"+str(index)] = each_file
    return req_dict

DATAFRAME_DEVICE = pd.DataFrame()
DATAFRAME_DEVICE["Select Device"] = sorted(get_dictionary(csv_files).keys())


##########################################################################################
                         # Main APP LAYOUT #
##########################################################################################

app = dash.Dash()
server = app.server
app.config['suppress_callback_exceptions']=True

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

##########################################################################################
                         # BATTERY PERFORMANCE GRAPH HELPER FUNC TAB #
##########################################################################################

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
            values_list.append("Name: "+each[0]+", Dev: "+each[1]+", Net_Stat: "+str(each[2])+" Time: "+each[3]+", Net_type: "+each[4])

        data = go.Scatter(x=pd.Series(range(0, 20000, 1)),
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
            values_list.append("Name: "+each[0]+", Dev: "+each[1])

        data = go.Scatter(x=pd.Series(range(0, 20000, 1)),
                          text=values_list,
                          mode="markers",
                          name="Network Quality",
                          hoverinfo="text",
                          marker={"color":"blue"},
                          y=dataframe['Battery_Percentage'])
        return data


def read_csv_file(csv_file_name):
    """
    Return dataframe
    """
    df = pd.read_csv(csv_file_name)
    return df

def get_data_specific_date(dataframe):
    """
    seperates data and time stamp
    """
    list_of_date_time = dataframe['Time_Stamp'].values.tolist()
    only_date = []
    date_value = []
    for each in list_of_date_time:
        date, value = each.split("-")[0],each.split("-")[0].split("/")[0]
        only_date.append(date)
        date_value.append(value)
    dataframe["Date"] = only_date
    return dataframe, list(set(date_value))

def get_data_specific_time(dataframe):
    """
    seperates time from time stamp
    """
    list_of_date_time = dataframe["Time_Stamp"].values.tolist()
    only_time = []
    timeFormat = "%H%M%S"
    for each in list_of_date_time:
        time = str(datetime.strptime(each.split("-")[1],timeFormat)).split(" ")[1]
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
    dataframe = read_csv_file(csv_files[file_index])
    df, date_values = get_data_specific_date(dataframe)
    df_date = filter_on_date(df, date_values)
    df_date = get_data_specific_time(df_date)
    return df_date


def get_dataframe_for_plotting_recording(file_index):
    """
    Returns Data frame more suitable format for plotting recording mode files
    """
    dataframe = read_csv_file(csv_files1[file_index])
    df, date_values = get_data_specific_date(dataframe)
    df_date = filter_on_date(df, date_values)
    df_date = get_data_specific_time(df_date)
    return df_date

##########################################################################################
                         # CALLBACK FOR URL INPUT: DEVICE TRANSMISSION #
##########################################################################################


# Update page
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    """
    Condition check for menu selection
    """
    if pathname == "/acoustic-device-report/transmission-performance":
        return html.Div(
            [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Select Devices"], className="subtitle padded"
                                    ),
                                    html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-transmission',
                                                                             columns=[{"name": i,
                                                                                       "id": i,
                                                                                       "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                             data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                             sorting=True,
                                                                             row_selectable="multi",
                                                                             style_table={"maxHeight":"200px",
                                                                                          "maxWidth" :"200px"})]),
                                             ], style={"margin-bottom":"10px"}
                                            ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        ["Device Transmission Details"],
                                        className="subtitle padded",
                                    ),
                                    html.Table(make_dash_table(DF_HIST_PRICES)),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(id="inside-transmission")

                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
            ],
            className="page",
        )


##########################################################################################
                         # CALLBACK FOR URL INPUT: LOCATION #
##########################################################################################


    elif pathname == "/acoustic-device-report/location-details":
        return html.Div(
            [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Select Devices"], className="subtitle padded"
                                    ),
                            html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-location',
                                                                     columns=[{"name": i,
                                                                               "id": i,
                                                                               "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                     data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                     sorting=True,
                                                                     row_selectable="multi",
                                                                     style_table={"maxHeight":"200px",
                                                                                  "maxWidth" :"200px"})]),
                                     ], style={"margin-bottom":"10px"}),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        ["Device Location Details"],
                                        className="subtitle padded",
                                    ),
                                    html.Table(make_dash_table(DF_FUND_FACTS)),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(id="inside-location")

                        ],
                        className="row ",
                    ),


                ],
                className="sub_page",
            ),
        ],
        className="page",
    )


##########################################################################################
                    # CALLBACK FOR URL INPUT: BATTERY PERFORMANCE #
##########################################################################################

    elif pathname == "/acoustic-device-report/battery-performance":
        return html.Div(
            [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Select Devices"], className="subtitle padded"
                                    ),
                            html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-battery',
                                                                     columns=[{"name": i,
                                                                                "id": i,
                                                                                "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                     data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                     sorting=True,
                                                                     row_selectable="multi",
                                                                     style_table={"maxHeight":"200px",
                                                                                  "maxWidth" :"200px"})]),
                                     ], style={"margin-bottom":"10px"}),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        ["Battery Performance"],
                                        className="subtitle padded",
                                    ),
                                    html.Table(make_dash_table(DF_HIST_PRICES)),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(id="inside-battery")

                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.A(id="my-link")

                        ],
                        className="row ",
                    ),

                ],
                className="sub_page",
            ),
        ],
        className="page",
    )


##########################################################################################
                    # CALLBACK FOR URL INPUT: DEVICE DETAILS #
##########################################################################################


    elif pathname == "/acoustic-device-report/device-details":
        return html.Div(
            [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Select Devices"], className="subtitle padded"
                                    ),
                            html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-device-details',
                                                                     columns=[{"name": i,
                                                                               "id": i,
                                                                               "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                     data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                     sorting=True,
                                                                     row_selectable="single",
                                                                     style_table={"maxHeight":"200px",
                                                                                  "maxWidth" :"200px"})]),
                                     ],style={"margin-bottom":"10px"}),
                                ],
                                className="six columns",
                            ),
                            html.Div(id="device-details-id",

                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(id="inside-device-details")

                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )


    elif pathname == "/acoustic-device-report/reviews":
        return html.Div(
            [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Select Devices"], className="subtitle padded"
                                    ),
                            html.Div([html.Div([dash_table.DataTable(id='datatable-interactivity-device-details',
                                                                     columns=[{"name": i,
                                                                                "id": i,
                                                                                "deletable": True} for i in DATAFRAME_DEVICE.columns],
                                                                     data=DATAFRAME_DEVICE.to_dict("rows"),
                                                                     sorting=True,
                                                                     row_selectable="single",
                                                                     style_table={"maxHeight":"200px",
                                                                                   "maxWidth" :"200px"})]),
                                     ], style={"margin-bottom":"10px"}),
                                ],
                                className="six columns",
                            ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Device Transmission Performance Review", className="subtitle padded",style={"margin-left":"5%"}),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.P(
                                                "Start Time of the Device:    17:59:03 - 19/07/2019",style={"margin-left":"5%"}
                                            ),
                                            html.P(
                                                "Estimated Recorded Files:   3600 (.wav) Files. ",style={"margin-left":"5%"}
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="six columns",
                            ),

                            html.Div(
                                [
                                    html.H6("Network Related", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.P(
                                                "TYpe of Network Quality"
                                            ),
                                            html.P(
                                                "Consistency of the Network Quality"
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                style={"margin-top":"5%",
                                        "margin-right":"-20%"},
                                className="six columns",
                            ),

                            html.Div(
                                [
                                    html.H6("Summary", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.P(
                                                ""
                                            ),
                                            html.P(
                                                ""
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                style={"margin-top":"5%"},
                                className="six columns",
                            ),

                        ],
                        className="row ",
                    )

                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(id="inside-device-details")

                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )



    elif pathname == "/acoustic-device-report/full-view":
        return overview.create_layout(app),
    else:
        return overview.create_layout(app)




##########################################################################################
                             # TRANSMISSION HELPER FUNCTION #
##########################################################################################

def plot_function_bar(dataframe):
    """
    Bar plot for transmision file details
    """
    print dataframe.head()
    val = dataframe[["Filename", "DeviceID", "Network_Status","Time_Stamp", "Network_Type"]].values
    values_list = []
    for each in val:
        values_list.append("Name: "+each[0]+", Dev: "+each[1]+", Net_Stat: "+str(each[2])+" Time: "+each[3]+", Net_type: "+each[4])

    trace1 = go.Bar(
        x=[dataframe["DeviceID"].iloc[0]],
        y=[dataframe.shape[0]],
        name= "Duration:- "+str(get_time_difference(dataframe["Time_Stamp"].iloc[0],dataframe["Time_Stamp"].iloc[dataframe.shape[0]-1])),
        text="No. Files:- "+str(dataframe.shape[0]),
        textposition="outside",
    )
    return trace1

##########################################################################################
                         # TIME STAMP DIFFERENCE HELPER FUNC #
##########################################################################################

def get_time_difference(timestamp1, timestamp2):
    """
    Returns the difference of time stamps
    """
    datetimeFormat = '%d/%m/%Y-%H%M%S'
    time_diff = datetime.strptime(timestamp2, datetimeFormat) - datetime.strptime(timestamp1, datetimeFormat)
    print "\ntime_diff :", str(time_diff)
    return time_diff


##########################################################################################
                         # LOCATION HELPER FUNC  #
##########################################################################################
latitudes=[]
longitudes=[]
device_name_location = []

mapbox_access_token = "pk.eyJ1IjoicHJpeWF0aGFyc2FuIiwiYSI6ImNqbGRyMGQ5YTBhcmkzcXF6YWZldnVvZXoifQ.sN7gyyHTIq1BSfHQRBZdHA"


def get_data(name):
    global latitudes, longitudes
    data = go.Scattermapbox(
        mode='markers',
        lat=latitudes,
        lon=longitudes,
        name=name,
        text=device_name_location,
        subplot='mapbox'
    )
    return data


def get_layout():

    layout = go.Layout(
            height=400, width=690,
            hovermode='closest',
            margin={
                "r": 0,
                "t": 0,
                "b": 0,
                "l": 0,
            },

            mapbox=go.layout.Mapbox(
                accesstoken=mapbox_access_token,
                domain={'x': [0, 1.0], 'y': [0, 1]},
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=float(latitudes[-1]),
                    lon=float(longitudes[-1])
                ),
                pitch=0,
                zoom=7))


    return layout


def get_figure(list_of_devices):

    global latitudes, longitudes, color_location, device_name_location   
    trace = []
    for index,each in enumerate(list_of_devices):
        df = read_csv_file(csv_files[int(str(each).split("_")[-1])])
        latitudes.append(df["Latitude"].iloc[0])
        longitudes.append(df["Longitude"].iloc[0])
        data = get_data(each)
        trace.append(data)
        layout = get_layout()                      
    fig = go.FigureWidget(data=trace, layout=layout)
    return fig


##########################################################################################
                         # CALLBACK FOR TRANSMISSION #
##########################################################################################



@app.callback(
    dash.dependencies.Output("inside-transmission", "children"),
    [Input('datatable-interactivity-transmission', 'data'),
     Input('datatable-interactivity-transmission', 'columns'),
     Input("datatable-interactivity-transmission", "derived_virtual_selected_rows")])
def update_figure_transmission(rows,columns,indices):
    """
    Callback for plots on  Transmission files details
    """
    fig = []
    req_dict = get_dictionary(csv_files)
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        print pred_df.values.tolist()
        selected = pred_df.values.tolist()
        if selected:
            if len(selected)==1:
                df_date = get_dataframe_for_plotting_transmission(int(str(selected[0]).split("_")[-1]))
                device1 = plot_function_bar(df_date)
                fig.append(device1)
            else:
                for each_fig in selected:
                    df_date = get_dataframe_for_plotting_transmission(int(str(each_fig).split("_")[-1]))
                    device1 = plot_function_bar(df_date)
                    fig.append(device1)
            return html.Div([html.H6("Device Transmission Performances", className="subtitle padded"),
                            dcc.Graph(figure={"data":fig,
                                             "layout": go.Layout(
                                                autosize=True,
                                                width=750,
                                                height=450,
                                                font={"family": "Raleway", "size": 10},
                                                margin={
                                                    "r": 0,
                                                    "t": 0,
                                                    "b": 45,
                                                    "l": 55,
                                                },
                                                showlegend=True,
                                                titlefont={
                                                    "family": "Raleway",
                                                    "size": 10,
                                                },
                                                xaxis={
                                                  'title': 'Device ID',
                                                  'titlefont':{
                                                      'family':'Courier New, monospace',
                                                      'size':16,
                                                      'color':'black'}
                                                },
                                                yaxis={
                                                  'title': 'No. Files Transmitted',
                                                  'titlefont':{
                                                      'family':'Courier New, monospace',
                                                      'size':16,
                                                      'color':'black'}
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                        style={"margin-top":"20px","margin-bottom":"10px"},
                                    ),
                                ],
                                className="twelve columns",style={"margin-left":"-57%","margin-top":"20%"})


##########################################################################################
                         # CALLBACK FOR LOCATION #
##########################################################################################
@app.callback(
    dash.dependencies.Output("inside-location", "children"),
    [Input('datatable-interactivity-location', 'data'),
     Input('datatable-interactivity-location', 'columns'),
     Input("datatable-interactivity-location", "derived_virtual_selected_rows")])
def update_figure_location(rows,columns,indices):
    """
    Callback for Location Plot
    """
    global latitudes, longitudes
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        print pred_df.values.tolist()
        selected = pred_df.values.tolist()
        if selected:
            fig = get_figure(selected)
            return html.Div([html.H6("Device Location Plot", className="subtitle padded"),
                            dcc.Graph(figure={"data":fig["data"],
                                            "layout":fig["layout"]},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",style={"margin-left":"-57%","margin-top":"20%"})

##########################################################################################
                          # CALLBACK FOR BATTERY PERFORMANCE GRAPH #
##########################################################################################

@app.callback(
    dash.dependencies.Output("inside-battery", "children"),
    [Input('datatable-interactivity-battery', 'data'),
     Input('datatable-interactivity-battery', 'columns'),
     Input("datatable-interactivity-battery", "derived_virtual_selected_rows")])
def update_figure_battery(rows,columns,indices):
    """
    Callback for Battery Plot
    """
    fig = []
    req_dict = get_dictionary(csv_files)
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        print pred_df.values.tolist()
        selected = pred_df.values.tolist()
        if selected:
            if len(selected)==1:
                df_date = get_dataframe_for_plotting_transmission(int(str(selected[0]).split("_")[-1]))
                # df_date = get_dataframe_for_plotting_recording(int(str(selected[0]).split("_")[-1]))
                device1 = plot_function(df_date)
                fig.append(device1)
            else:
                for each_fig in selected:
                    df_date = get_dataframe_for_plotting_transmission(int(str(each_fig).split("_")[-1]))
                    # df_date = get_dataframe_for_plotting_recording(int(str(selected[0]).split("_")[-1]))
                    device1 = plot_function(df_date)
                    fig.append(device1)
            return html.Div([html.H6("Device Battery Performance", className="subtitle padded"),
                             dcc.Graph(figure={"data":fig,
                                               "layout": go.Layout(
                                                   autosize=True,
                                                   width=750,
                                                   height=400,
                                                   font={"family": 'Courier New, monospace', "size": 10},
                                                   margin={"r": 30,
                                                           "t": 0,
                                                           "b": 35,
                                                           "l": 35,
                                                          },
                                                   showlegend=True,
                                                   titlefont={
                                                       "family": "Raleway",
                                                       "size": 10
                                                            },
                                                    xaxis={'title': 'Time ( Seconds )',
                                                           'titlefont':{'family':'Courier New, monospace',
                                                                        'size':16,
                                                                        'color':'black'}},
                                                    yaxis={'title': 'Battery Level ( Percentage )',
                                                           'titlefont':{'family':'Courier New, monospace',
                                                                        'size':16,
                                                                        'color':'black'}})},
                                        config={"displayModeBar": False},
                                        style={"margin-top":"20px","margin-bottom":"10px"}
                                    ),
                                ],
                                className="twelve columns", style={"margin-left":"-59%","margin-top":"30%"})



##########################################################################################
                         # CALLBACK FOR DEVICE DETAILS #
##########################################################################################
@app.callback(
    dash.dependencies.Output("device-details-id", "children"),
    [Input('datatable-interactivity-device-details', 'data'),
     Input('datatable-interactivity-device-details', 'columns'),
     Input("datatable-interactivity-device-details", "derived_virtual_selected_rows")])
def update_figure_device_details(rows,columns,indices):
    global latitudes, longitudes
    pred_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if indices is not None and indices != []:
        pred_df = pred_df.iloc[indices]["Select Device"]
        print pred_df.values.tolist()
        selected = pred_df.values.tolist()
        if selected:
            df = pd.read_csv(csv_files[int(str(selected[0]).split("_")[-1])])
            DF_FUND_FACTS = pd.DataFrame()
            DF_FUND_FACTS["label"] = df.columns
            DF_FUND_FACTS["value"] = df.iloc[1,:].values.tolist()
            print DF_FUND_FACTS.head()
            return  [html.H6(["Device Details - "+selected[0]],
                                        className="subtitle padded",
                                    ),
                                    html.Table(make_dash_table(DF_FUND_FACTS)),]



##########################################################################################
                         # CALLBACK FOR DOWNLOAD REPORT #
##########################################################################################



if __name__ == "__main__":
    app.run_server(debug=True)
