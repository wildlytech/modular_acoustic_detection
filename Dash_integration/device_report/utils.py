import dash_html_components as html
import dash_core_components as dcc


###############################################################################
# consolidation of the header
###############################################################################
def Header(app):
    """
    consolidation of the header
    """
    return html.Div([get_header(app), html.Br([]), get_menu()])


###############################################################################
# Returns the header of the app
###############################################################################
def get_header(app):
    """
    Returns the header of the app
    """
    header = html.Div(
        [html.Div(
            [html.Img(src=app.get_asset_url("dash-logo.png"),
                      className="logo"),
             html.A(html.Button("Learn More",
                                id="learn-more-button",
                                n_clicks=0),
                    href="/acoustic-device-report/summary")],
            className="row"),
         html.Div(
             [html.Div(
                 [html.H5("Acoustic Monitoring Device Report",
                          style={'text-decoration': 'underline'})
                  ], className="seven columns main-title"),
              html.Div(
                  [html.A("Download Report",
                          id="download-link-report",
                          download='readMe.pdf',
                          style={'text-decoration': 'underline'},
                          href="/readMe.pdf",
                          className="full-view-link")
                   ], className="five columns")
              ], className="twelve columns",
             style={"padding-left": "0"})
         ], className="row")

    return header


###############################################################################
# Menu options
###############################################################################
def get_menu():
    """
    Menu options
    """
    menu = html.Div(
        [
            dcc.Link(
                "Monitor Device",
                href="/acoustic-device-report/summary",
                className="tab first",
            ),
            dcc.Link(
                "Transmission Performance",
                href="/acoustic-device-report/transmission-performance",
                className="tab",
            ),
            dcc.Link(
                "Location Details",
                href="/acoustic-device-report/location-details",
                className="tab",
            ),
            dcc.Link(
                "Battery Report", href="/acoustic-device-report/battery-performance", className="tab"
            ),
            dcc.Link(
                "Device Detail",
                href="/acoustic-device-report/device-details",
                className="tab",
            ),
            dcc.Link(
                "Reviews",
                href="/acoustic-device-report/reviews",
                className="tab",
            ),
            dcc.Link(
                "Overview",
                href="/acoustic-device-report/overview",
                className="tab",
            )
        ],
        className="row all-tabs",
    )
    return menu


###############################################################################
# Return a dash definition of an HTML table for a Pandas dataframe
###############################################################################
def make_dash_table(dataframe):
    """
    Return a dash definition of an HTML table for a Pandas dataframe
    """
    table = []
    for _, row in dataframe.iterrows():
        html_row = []
        for index in range(len(row)):
            html_row.append(html.Td([row[index]]))
        table.append(html.Tr(html_row))
    return table
