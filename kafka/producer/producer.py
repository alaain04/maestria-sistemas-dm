#!/usr/bin/env python
# coding: utf-8
from time import sleep
from json import dumps
from kafka import KafkaProducer
import pandas as pd

#Constants
kafka_server = 'kafka:9092'
topic = 'orders-topic'

def run_producer():
    producer = KafkaProducer(
                            bootstrap_servers=[kafka_server],
                            value_serializer=lambda x: dumps(x).encode('utf-8')
                        )
    # Cargamos todos los csv en df diferentes
    olist_orders = pd.read_csv('./dataset/olist_orders_dataset.csv')
    olist_products = pd.read_csv('./dataset/olist_products_dataset.csv')
    olist_items = pd.read_csv('./dataset/olist_order_items_dataset.csv')
    olist_customers = pd.read_csv('./dataset/olist_customers_dataset.csv')
    olist_payments = pd.read_csv('./dataset/olist_order_payments_dataset.csv')
    olist_sellers = pd.read_csv('./dataset/olist_sellers_dataset.csv')
    olist_geolocation = pd.read_csv('./dataset/olist_geolocation_dataset.csv')
    olist_reviews = pd.read_csv('./dataset/olist_order_reviews_dataset.csv')
    olist_product_category_name = pd.read_csv('./dataset/product_category_name_translation.csv')

    # Mergeamos los df anteriores en uno solo
    all_data = olist_orders.merge(olist_items, on='order_id', how='left')
    all_data = all_data.merge(olist_payments, on='order_id', how='inner')
    all_data = all_data.merge(olist_reviews, on='order_id', how='inner')
    all_data = all_data.merge(olist_products, on='product_id', how='inner')
    all_data = all_data.merge(olist_customers, on='customer_id', how='inner')
    all_data = all_data.merge(olist_sellers, on='seller_id', how='inner')
    all_data = all_data.merge(olist_product_category_name,on='product_category_name',how='inner')

    all_data.to_json()

    for index, row in all_data.iterrows():
        if(index%10==0):
            print("Producer has pushed {} messages".format(index))
        data = row.to_json()    
        producer.send(topic, value=data)
        sleep(1)


try:
    print('*'*100 + '\nStarting producer\n' + '*'*100)
    run_producer()
except Exception:
    print('Waiting kafka fulfil start up...')
    sleep(2)
    quit()
