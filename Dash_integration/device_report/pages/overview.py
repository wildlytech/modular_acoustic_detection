import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import glob
from ..utils import Header, make_dash_table
import pandas as pd
import numpy as np
from datetime import datetime

csv_files = glob.glob("transmission_details/*.csv")


###############################################################################
# BATTERY PERFORMANCE GRAPH HELPER FUNC TAB
###############################################################################

def SetColor(x):
    if(x > 20):
        return "green"
    elif(x > 15 and x <= 20):
        return "yellow"
    elif(x >= 10 and x <= 15):
        return "red"
    elif(x < 10):
        return "red"


def plot_function(dataframe):

    try:
        val = dataframe[["Filename", "DeviceID", "Network_Status", "Time_Stamp", "Network_Type"]].values
        values_list = []
        for each in val:
            values_list.append("Name: " + each[0] + ", Dev: " + each[1] + ", Net_Stat: " + str(each[2]) + " Time: " + each[3] + ", Net_type: " + each[4])

        values_color_dict = dict(color=list(map(SetColor, dataframe['Network_Status'])))
        values_color_list = values_color_dict["color"]
        unique_colors, counts = np.unique(np.array(values_color_list), return_counts=True)
        names_counts_dict = dict(list(zip(unique_colors, counts)))
        if len(list(names_counts_dict.values())) > 1:
            max_value = max(names_counts_dict.values())
            get_index = list(names_counts_dict.values()).index(max_value)
            color_main = list(names_counts_dict.keys())[get_index]
            print(color_main)
            if color_main == "red":
                name = "Network Quality:- BAD"
            elif color_main == "yellow":
                name = "Network Quality:- AVERAGE"
            elif color_main == "green":
                name = "Network Quality:- GOOD"
            else:
                pass
        else:
            color_main = list(names_counts_dict.keys())[0]
            if color_main == "red":
                name = "Network Quality:- BAD"
            elif color_main == "green":
                name = "Network Quality:- GOOD"
            elif color_main == "yellow":
                name = "Network Quality:- AVERAGE"
        data = go.Scatter(x=pd.Series(list(range(0, 20000, 1))), text=values_list, mode="markers", hoverinfo="text",
                            name=name, marker=dict(color=list(map(SetColor, dataframe['Network_Status']))),
                            y=dataframe['Battery_Percentage'])
        return data
    except TypeError:
        val = dataframe[["Filename", "DeviceID"]].values
        values_list = []
        for each in val:
            values_list.append("Name: " + each[0] + ", Dev: " + each[1])
        values_color = set(dict(color=list(map(SetColor, dataframe['Network_Status']))).keys())
        if len(list(values_color)) == 1:
            if list(values_color)[0] == "red":
                name = "Network Quality:- " + list(values_color)[0] + "=BAD"
            else:
                name = "Network Quality:- " + list(values_color)[0] + "=GOOD"
        else:
            name = "Network Quality"
        data = go.Scatter(x=pd.Series(list(range(0, 10000, 1))), text=values_list, mode="markers", name="Network Quality", hoverinfo="text",
                       marker={"color": "blue"}, y=dataframe['Battery_Percentage'])
        return data


###############################################################################
# TIME STAMP DIFFERENCE HELPER FUNC
###############################################################################

def get_time_difference(timestamp1, timestamp2):

    datetimeFormat = '%d/%m/%Y-%H%M%S'
    time_diff = datetime.strptime(timestamp2, datetimeFormat) - datetime.strptime(timestamp1, datetimeFormat)
    return time_diff


###############################################################################
# Device Transmission HELPER FUNC
###############################################################################

def plot_function_bar(dataframe, device_value):
    # print dataframe.head()
    val = dataframe[["Filename", "DeviceID", "Network_Status", "Time_Stamp", "Network_Type"]].values
    values_list = []
    for each in val:
        values_list.append("Name: " + each[0] + ", Dev: " + each[1] + ", Net_Stat: " + str(each[2]) + " Time: " + each[3] + ", Net_type: " + each[4])

    trace1 = go.Bar(
        x=[dataframe["DeviceID"].iloc[0]],
        y=[dataframe.shape[0]],
        name="Device-" + str(device_value),
        textposition="outside",
        marker=dict(
        line=dict(
            color='red',
            width=0.01),
        )
    )
    return trace1


def read_csv_file(csv_file_name):
    df = pd.read_csv(csv_file_name)
    return df


def get_data_specific_date(dataframe):
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
    list_of_date_time = dataframe["Time_Stamp"].values.tolist()
    only_time = []
    timeFormat = "%H%M%S"
    for each in list_of_date_time:
        time = str(datetime.strptime(each.split("-")[1], timeFormat)).split(" ")[1]
        only_time.append(time)

    dataframe["Time"] = only_time
    return dataframe


def filter_on_date(dataframe, date):
    return dataframe.loc[dataframe["Date"].apply(lambda arr: arr.split("/")[0] == date[0] or arr.split("/")[0] == date[1])]


def get_dataframe_for_plotting_transmission(file_index):
    dataframe = read_csv_file(csv_files[file_index])
    df, date_values = get_data_specific_date(dataframe)
    df_date = filter_on_date(df, date_values)
    df_date = get_data_specific_time(df_date)
    return df_date


def get_dictionary(csv_files):
    req_dict = dict()
    for index, each_file in enumerate(csv_files):
        req_dict["Device_" + str(index)] = each_file
    return req_dict


###############################################################################
# LOCATION HELPER FUNC
###############################################################################
latitudes = []
longitudes = []
color_location = []
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
            height=360, width=390,
            hovermode='closest',
            margin={
                "r": 0,
                "t": 0,
                "b": 0,
                "l": 10,
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
                zoom=5))

    return layout


def get_figure(list_of_devices):

    global latitudes, longitudes, color_location, device_name_location
    trace = []
    for index, each in enumerate(list_of_devices):
        df = read_csv_file(csv_files[int(str(each).split("_")[-1])])
        latitudes.append(df["Latitude"].iloc[0])
        longitudes.append(df["Longitude"].iloc[0])
        # color_location.append("red")
        device_name_location.append(each)
        data = get_data(each)
        trace.append(data)
        layout = get_layout()
    fig = go.FigureWidget(data=trace, layout=layout)
    return fig


###############################################################################
# LAYOUT FUNC
###############################################################################

def create_layout(app):
    fig_battery = []
    fig_location = []
    fig_device_trans = []

    df = pd.read_csv(csv_files[0])
    df_fund_facts = pd.DataFrame()
    df_fund_facts["label"] = df.columns
    df_fund_facts["value"] = df.iloc[1, :].values.tolist()
    # fig = get_dataframe_for_plotting_transmission()
    selected_device_trans = [0, 1, 2]
    for each_fig in selected_device_trans:
        df_date = get_dataframe_for_plotting_transmission(each_fig)
        device1 = plot_function_bar(df_date, each_fig)
        fig_device_trans.append(device1)
    selected_battery = [0]
    for each_fig in selected_battery:
        df_date = get_dataframe_for_plotting_transmission(each_fig)
        device1 = plot_function(df_date)
        # fig_battery = device1
        fig_battery.append(device1)
    selected_location = [0]
    # for each_fig in selected_location:
    fig_location = get_figure(selected_location)
    data_location = fig_location['data']

    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Product Summary"),
                                    html.Br([]),
                                    html.P("Wildly Listen Product Specifications and Operations in detail",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Device Details"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_fund_facts)),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Transmission Performance",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-1",
                                        figure={"data": fig_device_trans,
                                            "layout": dict(
                                                autosize=False,
                                                bargap=0.4,
                                                font={"family": "Courier New, monospace", "size": 10},
                                                height=500,
                                                hovermode="closest",
                                                legend={
                                                    "x": -0.0228945952895,
                                                    "y": -0.189563896463,
                                                    "orientation": "h",
                                                    "yanchor": "top",
                                                },
                                                margin={
                                                    "r": 10,
                                                    "t": 40,
                                                    # "b": 35,
                                                    # "l": 45,
                                                },
                                                showlegend=False,
                                                # title="Transmission",
                                                width=350,
                                                xaxis={
                                                  'title': 'Device ID',
                                                  'titlefont': {
                                                      'family': 'Courier New, monospace',
                                                      'size': 16,
                                                      'color': 'black'}
                                                },
                                                yaxis={
                                                  'title': 'No. Files',
                                                  'titlefont': {
                                                      'family': 'Courier New, monospace',
                                                      'size': 16,
                                                      'color': 'black'}
                                                })
                                            },

                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Battery Performance",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-2",
                                        figure={"data": fig_battery,

                                            "layout": go.Layout(
                                                autosize=True,
                                                font={"family": 'Courier New, monospace', "size": 8},
                                                height=400,
                                                width=350,
                                                hovermode="closest",
                                                legend={
                                                    "x": -0.0277108433735,
                                                    "y": -0.142606516291,
                                                    "orientation": "h",
                                                },
                                                margin={
                                                    "r": 0,
                                                    "t": 0,
                                                    "b": 35,
                                                    "l": 45,
                                                },
                                                showlegend=False,
                                                xaxis={
                                                  'title': 'Time (Seconds) ',
                                                  'titlefont': {
                                                      'family': 'Courier New, monospace',
                                                      'size': 16,
                                                      'color': 'black'}
                                                },
                                                yaxis={
                                                  'title': 'Battey Level (Percentage) ',
                                                  'titlefont': {
                                                      'family': 'Courier New, monospace',
                                                      'size': 16,
                                                      'color': 'black'}
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                        style={"margin-bottom": "10px"}
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Location",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-3",
                                        figure={"data": data_location,
                                            "layout": fig_location["layout"]},
                                        config={"displayModeBar": False},
                                    ),
                                ], style={"margin-right": "-20%", "margin-bottom": "10px"},
                                className="six columns",
                            ),

                            html.Div(
                                [
                                    html.H6(
                                        "Device Health Status", className="subtitle padded"
                                    ),
                                    html.Img(
                                        src=app.get_asset_url("risk_reward.png"),
                                        className="risk-reward",
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
