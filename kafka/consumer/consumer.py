#!/usr/bin/env python
# coding: utf-8
from kafka import KafkaConsumer
from json import loads
from time import sleep
import pandas as pd

#Constants
kafka_server = 'kafka:9092'
topic = 'orders-topic'

def run_consumer():
    consumer = KafkaConsumer(
                            topic, 
                            bootstrap_servers=[kafka_server],
                            auto_offset_reset='earliest',
                            enable_auto_commit=True,
                            group_id='test-group'
                        )
    for message in consumer:
        try:
            data = loads(message.value.decode('utf-8'))
            df = pd.DataFrame([data])
            df.to_csv('test.csv')
            print('{}'.format(df.head()))
        except Exception:
          print('error')
          continue


try:
    print('*'*100 + '\nStarting consumer\n' + '*'*100)
    run_consumer()
except Exception:
    print('Waiting kafka fulfil start up...')
    sleep(2)
    quit()



