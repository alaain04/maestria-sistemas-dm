#!/usr/bin/env python
# coding: utf-8
from kafka import KafkaConsumer
from json import loads
from time import sleep
import pandas as pd
import requests

# Constants
kafka_server = 'kafka:9092'
topic = 'orders-topic'
server_url = 'http://kmeans:5000'
batch_size = 100

def run_consumer():
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[kafka_server],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='test-group'
    )
    batch_data = []
    headers = {'Content-type': 'application/json',
               'Accept': 'text/plain'}
    print('Connected to the broker ..')
    for message in consumer:
        try:

            data = loads(message.value.decode('utf-8'))
            batch_data.append(data)
            if(len(batch_data) == batch_size):
                try:
                    requests.post(server_url + '/data', headers=headers, json=batch_data)
                except Exception as e:
                    print('Error',e)
                batch_data = []
        except Exception as e:
            print('Error', e)


try:
    print('*'*100 + '\nStarting consumer\n' + '*'*100)
    run_consumer()
except Exception:
    print('Waiting kafka fulfil start up...')
    sleep(2)
    quit()
