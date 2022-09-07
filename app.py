from dash import Dash, html, dcc, Input, Output, State
from summarizers.SumBasicSummarizer import SumBasicSummarizer
from summarizers.TFIDFSummarizer import TFIDFSummarizer
from summarizers.TextRankSummarizer import TextRankSummarizer
from summarizers.LexRankSummarizer import LexRankSummarizer
from summarizers.AbstractiveSummarizer import AbstractiveSummarizer

app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

print("Loading...")
summarizers = {
    "SumBasic": SumBasicSummarizer(),
    "TF-IDF": TFIDFSummarizer(),
    "TextRank": TextRankSummarizer(),
    "LexRank": LexRankSummarizer(),
    "Algorytm abstrakcyjny": AbstractiveSummarizer()
}
print("Loaded.")

app.layout = html.Div(
    children=[
        # Title
        html.H1(
            children='Automatyczne podsumowywanie tekstów w języku polskim',
            style={
                "textAlign": "center"
            }
        ),

        html.Div(
            children=[
                # Input
                html.Div(
                    children=[
                        html.H3(children="Tekst do podsumowania"),

                        dcc.Textarea(
                            id='input-textarea',
                            placeholder='Wprowadź tekst...',
                            draggable=False,
                            style={
                                'width': '100%',
                                'height': 650
                            }
                        ),
                    ],
                    style={
                        'display': 'inline-block',
                        'width': '47%',
                        'margin-left': '1%',
                        'margin-right': '1%'
                    }
                ),

                # Settings and output
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.H3(children='Algorytm'),
                                        dcc.RadioItems(
                                            id='algorithm-select-radio',
                                            options=list(summarizers.keys()),
                                            value=list(summarizers.keys())[0],
                                            labelStyle={
                                                'display': 'block',
                                                'margin-top': '5px'
                                            }
                                        )
                                    ],
                                    style={
                                        'display': 'inline-block'
                                    }
                                ),
                                html.Div(
                                    children=[
                                        html.H3(children='Rozmiar podsumowania'),
                                        html.Div(
                                            id='size-description'
                                        ),
                                        dcc.Input(
                                            id='n-input',
                                            type='number',
                                            value=5
                                        ),
                                        html.Button(
                                            id='summarize-button',
                                            children='Podsumowanie',
                                            style={
                                                'display': 'block',
                                                'margin-top': '15px'
                                            }
                                        ),
                                    ],
                                    style={
                                        'display': 'inline-block',
                                        'margin-left': '50px',
                                        'verticalAlign': 'top',
                                    }
                                )
                            ],
                            style={
                                'margin-left': '30px'
                            }
                        ),
                        html.H3(children="Podsumowanie"),
                        dcc.Textarea(
                            id='result-textarea',
                            style={
                                'width': '100%',
                                'height': 440
                            }
                        ),
                    ],
                    style={
                        'display': 'inline-block',
                        'width': '47%',
                        'verticalAlign': 'top',
                        'margin-left': '1%',
                        'margin-right': '1%'
                    }
                ),
            ]
        ),
    ],
    style={
        'margin-top': '30px',
        'margin-left': '5%',
        'margin-right': '5%'
    }
)


@app.callback(
    Output('result-textarea', 'value'),
    Input('summarize-button', 'n_clicks'),
    [State('algorithm-select-radio', 'value'),
     State('input-textarea', 'value'),
     State('n-input', 'value')],
    prevent_initial_call=True
)
def summarize(n_clicks, algorithm, text, n):
    print("summarize")
    return summarizers[algorithm].summarize(text, n)


@app.callback(
    Output('size-description', 'children'),
    Input('algorithm-select-radio', 'value')
)
def size_description(algorithm):
    return "Procent tekstu wejściowego" if algorithm == "Algorytm abstrakcyjny" else "Liczba zdań"


if __name__ == '__main__':
    app.run_server(debug=True)
