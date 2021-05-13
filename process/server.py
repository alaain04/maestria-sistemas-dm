from flask import Flask, request
from json import dumps, loads
from etl import Etl
from clustering import Clustering
import pandas as pd
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
server_url = 'http://dashboard:8050'


@app.route('/data', methods=['POST'])
def process_data():
    data = request.get_data()
    data = loads(data)
    cluster_data(data)
    return dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/test', methods=['GET'])
def test():
    print('Connected')
    return dumps({'success': True}), 200, {'ContentType': 'application/json'}


def cluster_data(data):
    etl = Etl()
    df = etl.process_data(data)
    df = etl.generate_rfm(df)
    df = etl.normalize_df(df)
    clustering = Clustering()
    [metrics, clusters] = clustering.generate_cluster(df)
    headers = {'Content-type': 'application/json',
               'Accept': 'text/plain'}
    try:
        requests.post(server_url + '/metrics', headers=headers, json=metrics)
        requests.post(server_url + '/clusters', headers=headers, json=clusters)
    except Exception as e:
        print('Error', e)


app.run(debug=True, host='0.0.0.0', port=5000)
