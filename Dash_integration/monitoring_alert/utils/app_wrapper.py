import base64
import dash_core_components as dcc
import dash_html_components as html
import os


def app_page_layout(page_layout,
                    app_title="Dash Bio App",
                    app_name="",
                    light_logo=True,
                    standalone=False,
                    bg_color="#506784",
                    font_color="#F3F6FA"):
    return html.Div(
        id='main_page',
        children=[
            dcc.Location(id='url', refresh=False),
            html.Div(
                id='app-page-header',
                children=[
                    html.A(
                        id='dashbio-logo-graph', children=[
                        html.P("<< Main Page", style={"display": "inline",
                                                      "text-decoration": "underline"}),
                        ],
                        href="/Monitoring" if standalone else "/dash-bio"
                    ),
                    html.H2(
                        app_title,
                        style={"display": "inline",
                               "font-family": "Courier New",
                               "fontSize": "25px",
                               "margin-left": "7%",
                               "text-decoration": "underline"}
                    ),

                    html.A(
                        id='gh-link',
                        children=[
                            'View on GitHub'
                        ],
                        href="https://github.com/wildlytech",
                        style={'color': 'white' if light_logo else 'black',
                               'border': 'solid 1px white' if light_logo else 'solid 1px black'}
                    ),

                    html.Img(
                        src='data:image/png;base64,{}'.format(
                            base64.b64encode(
                                open(
                                    os.path.dirname(__file__) + '/../assets/GitHub-Mark-{}64px.png'.format(
                                        'Light-' if light_logo else ''
                                    ),
                                    'rb'
                                ).read()
                            ).decode()
                        )
                    )
                ],
                style={
                    'background': bg_color,
                    'color': font_color,
                }
            ),
            html.Div(
                id='app-page-content',
                children=page_layout
            )
        ],
    )
