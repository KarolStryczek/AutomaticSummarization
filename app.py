from dash import Dash, html, dcc, Input, Output, State
from summarizers.SumBasicSummarizer import SumBasicSummarizer
from summarizers.TFIDFSummarizer import TFIDFSummarizer
from summarizers.TextRankSummarizer import TextRankSummarizer
from summarizers.LexRankSummarizer import LexRankSummarizer
from summarizers.AbstractiveSummarizer import AbstractiveSummarizer

print("Loading...")
summarizers = {
    "SumBasic": SumBasicSummarizer(),
    "TF-IDF": TFIDFSummarizer(),
    "TextRank": TextRankSummarizer(),
    "LexRank": LexRankSummarizer(),
    "Algorytm abstrakcyjny": AbstractiveSummarizer()
}
print("Loaded.")


app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Automatyczne podsumowywanie tekstów w języku polskim'),

    html.Div(children='Wybierz algorytm'),

    dcc.RadioItems(
        id='algorithm-select-radio',
        options=list(summarizers.keys()),
        value=list(summarizers.keys())[0],
        labelStyle={'display': 'block'}
    ),

    html.Div(children='n:'),
    dcc.Input(
        id='n-input',
        type='number',
        value=5
    ),

    dcc.Textarea(
        id='input-textarea',
        placeholder='Wprowadź tekst...',
        style={
            'width': '100%',
            'height': 500
        }
    ),

    html.Button(id='summarize-button', children='Submit'),

    html.Div(id='result-textarea', style={'whiteSpace': 'pre-line'})
])


@app.callback(
    Output('result-textarea', 'children'),
    Input('summarize-button', 'n_clicks'),
    [State('algorithm-select-radio', 'value'),
     State('input-textarea', 'value'),
     State('n-input', 'value')]
)
def summarize(n_clicks, algorithm, text, n):
    if n_clicks is None:
        return ""
    else:
        return summarizers[algorithm].summarize(text, n)


if __name__ == '__main__':
    app.run_server(debug=True)
