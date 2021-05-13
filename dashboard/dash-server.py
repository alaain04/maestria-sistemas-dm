from logging import logProcesses
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask import Flask, request
from datetime import datetime, timedelta
from dash.dependencies import Input, Output
import pandas as pd
import json
import plotly.graph_objs as go
import plotly.express as px


server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

n_clusters = 5
n_predictions = 0
last_update = datetime.now()
tmp_cluster = 'tmp_clusters.csv'
tmp_metric = 'tmp_metric.csv'
tmp_data = 'tmp_data.csv'
response = json.dumps({'success': True}), 200, {
    'ContentType': 'application/json'}


@server.route('/clusters', methods=['POST'])
def clusters_data():
    global n_predictions
    n_predictions += 1
    data = request.get_data()
    data = json.loads(data)
    df = pd.DataFrame(data)
    df.to_csv(tmp_cluster)
    global last_update
    last_update = datetime.now()
    return response


@server.route('/metrics', methods=['POST'])
def metrics_data():
    data = request.get_data()
    data = json.loads(data)
    df = pd.DataFrame([data])
    df.to_csv(tmp_metric, index=False)
    return response


app.layout = dbc.Container(
    [
        html.H2('Incremental Learning - Kafka'),
        html.Div(id='id-cards'),
        html.H4('Clusters'),
        dcc.Graph(id='id-clusters'),

        dcc.Graph(id='id-metrics'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        ),
    ],
    fluid=True,
    style={'backgroundColor': '#aeaeae'}
)


@app.callback(Output('id-cards', 'children'),
              Input('interval-component', 'n_intervals'))
def update_cards(n):
    global last_update
    dt = last_update.strftime("%H:%M:%S")
    time_card = [
        dbc.CardHeader("Updated at"),
        dbc.CardBody(
            [
                html.H5(dt, className="card-title"),
            ]
        ),
    ]

    predictions_card = [
        dbc.CardHeader("Predictions"),
        dbc.CardBody(
            [
                html.H5(n_predictions, className="card-title"),
            ]
        ),
    ]
    clusters_card = [
        dbc.CardHeader("Clusters"),
        dbc.CardBody(
            [
                html.H5(n_clusters, className="card-title"),
            ]
        ),
    ]
    return dbc.Row(
        [
            dbc.Col(dbc.Card(time_card, color="secondary", outline=True)),
            dbc.Col(dbc.Card(predictions_card,
                             color="secondary", outline=True)),
            dbc.Col(dbc.Card(clusters_card,
                             color="secondary", outline=True)),
        ],
        className="mb-4",
    )


@app.callback(Output('id-metrics', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    df = pd.read_csv(tmp_metric)
    df = df.drop(['metric_bic_value'], axis=1)
    return go.Figure(data=[go.Bar(x=df.values[0],
                                  y=list(df.columns),
                                  orientation='h',
                                  )],
                     layout=go.Layout(title='Metrics', xaxis=dict(
                         title='Value'), yaxis=dict(title='Metrics'), hovermode='closest'))


@app.callback(Output('id-clusters', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_clusters(n):
    df_clusters = pd.read_csv(tmp_cluster)
    colorscale = [[0, 'red'], [0.5, 'gray'], [1.0, 'green']]
    data = [go.Scatter(
        x=df_clusters['monetary_value'],
        y=df_clusters['recency'],
        mode="markers",
        marker={'color': df_clusters.index,
                'colorscale': colorscale,
                "size": 12,
                "symbol": "star-dot"
                },
        # hover_data=df_clusters.index,
        name="Cluster".format()
    )]

    layout = {"xaxis": {"title": "Recency"},
              "yaxis": {"title":  "Monetary"}}

    return go.Figure(data=data, layout=layout,
                     layout_yaxis_range=[-4, 4],
                     layout_xaxis_range=[-4, 4])


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
