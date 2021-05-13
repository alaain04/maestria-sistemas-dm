#!/usr/bin/env python
# coding: utf-8

#Importing Libraries
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
# # 2. PREPROCESAMIENTO DE DATOS 

class Etl:
        
    def process_data(self, string_data):
        # Transformamos el array en un dataframe
        json_data = [json.loads(line) for line in string_data]
        all_data =pd.DataFrame(json_data)
        # Casteamos las siguientes columnas que contiene números a int64
        all_data = all_data.astype({'order_item_id': 'int64', 
                                    'product_name_lenght': 'int64',
                                    'product_description_lenght':'int64', 
                                    'product_photos_qty':'int64'})
        # Casteamos las columnas que contienen fechas a datetime
        date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                    'order_estimated_delivery_date', 'shipping_limit_date', 'review_creation_date', 'review_answer_timestamp'] 
        for col in date_columns:
            all_data[col] = pd.to_datetime(all_data[col], format='%Y-%m-%d %H:%M:%S')

        # ## 2.2 Manejo de valores perdidos
        # 
        # Esta etapa se hace para eliminar entradas vacías mediante el uso de otras características o el reemplazando este volor perdido por la media / mediana.

        # Gestion de las entradas vacías en la columna order_approved_at
        missing_1 = all_data['order_approved_at'] - all_data['order_purchase_timestamp']
        #print(missing_1.describe())
        print('Mediana desde el momento en que se aprobó la orden: ',missing_1.median())

        add_1 = all_data[all_data['order_approved_at'].isnull()]['order_purchase_timestamp'] + missing_1.median()
        all_data['order_approved_at'].fillna(add_1, inplace=True)

        # Gestion de las entradas vacías en la columna order_delivered_carrier_date
        missing_2 = all_data['order_delivered_carrier_date'] - all_data['order_approved_at']
        #print(missing_2.describe())
        print('Mediana desde el momento de la solicitud hasta el envío: ',missing_2.median())

        add_2 = all_data[all_data['order_delivered_carrier_date'].isnull()]['order_approved_at'] + missing_2.median()
        all_data['order_delivered_carrier_date'].fillna(add_2, inplace=True)

        # Gestion de las entradas vacías en la columna order_delivered_customer_date
        missing_3 = all_data['order_delivered_customer_date'] - all_data['order_delivered_carrier_date']
        #print(missing_3.describe())
        print('Mediana desde el momento en que se envió hasta que el cliente la recibió: ',missing_3.median())

        add_3 = all_data[all_data['order_delivered_customer_date'].isnull()]['order_delivered_carrier_date'] + missing_3.median()
        all_data['order_delivered_customer_date'].fillna(add_3, inplace=True)

        # El número de celdas en blanco en las columnas review_comment_title y review_comment_message es muy grande 
        # e imposible de completar. Entonces se eliminan.

        all_data = all_data.drop(['review_comment_title', 'review_comment_message'], axis=1)

        # El número de celdas en blanco en las columnas product_weight_g, product_length_cm, product_height_cm, product_width_cm
        # es solo una, entonces borramos estos registros
        all_data = all_data.dropna()

        # ## 2.3. Extracción de características 
        # 
        # En esta etapa crearemos nuevas caracteristicas a partir de las columnas existentes. 
        # 

        # Creamos una columna order_process_time para ver cuánto tiempo llevará iniciar el pedido hasta
        # que los artículos son aceptados por los clientes
        all_data['order_process_time'] = all_data['order_delivered_customer_date'] - all_data['order_purchase_timestamp']

        # Creamos una columna order_delivery_time para ver cuánto tiempo se requiere de envío para cada pedido
        all_data['order_delivery_time'] = all_data['order_delivered_customer_date'] - all_data['order_delivered_carrier_date']

        # Creamos una columna order_time_accuracy para ver la diferencia entre el tiempo estimado de envio y el tiempo que 
        # realmente demoro. Si el valor es positivo entonces es más rápido hasta, si es cero, está justo a tiempo 
        # y si es negativo llego tarde
        all_data['order_accuracy_time'] = all_data['order_estimated_delivery_date'] - all_data['order_delivered_customer_date'] 

        # Creamos una columna order_approved_time para ver cuánto tiempo tomará desde el pedido hasta la aprobación
        all_data['order_approved_time'] = all_data['order_approved_at'] - all_data['order_purchase_timestamp'] 

        # Creamos una columna review_send_time para averiguar cuánto tiempo se envió la encuesta de satisfacción después de recibir el artículo.
        all_data['review_send_time'] = all_data['review_creation_date'] - all_data['order_delivered_customer_date']

        # Creamos una columna review_answer_time 
        all_data['review_answer_time'] = all_data['review_answer_timestamp'] - all_data['review_creation_date']

        # Combinamos las columnas product_length_cm, product_height_cm y product_width_cm para convertirlo en un volumen
        all_data['product_volume'] = all_data['product_length_cm'] * all_data['product_height_cm'] * all_data['product_width_cm']

        # Creamos una columna month_order para la exploración de datos
        all_data['month_order'] = all_data['order_purchase_timestamp'].dt.to_period('M').astype('str')
        all_data[['month_order','order_purchase_timestamp']].head()

        # Nos quedamos con las columnas que van desde 01-2017 hasta 08-2018
        # Porque hay datos que están fuera de balance con el promedio de cada mes en los datos antes del 01-2017 
        # y después del 08-2018 basado en datos de compra / order_purchase_timestamp
        start_date = "2017-01-01"
        end_date = "2018-08-31"

        after_start_date = all_data['order_purchase_timestamp'] >= start_date
        before_end_date = all_data['order_purchase_timestamp'] <= end_date
        between_two_dates = after_start_date & before_end_date
        all_data = all_data.loc[between_two_dates]

        # # Datos para geovisualización
        # df_orders_items = olist_orders.merge(olist_customers, how='left', on='customer_id')

        # # Utilizamos API del gobierno de Brasil 
        # requests_result = requests.get('https://servicodados.ibge.gov.br/api/v1/localidades/mesorregioes')
        # uf = [mesorregion['UF'] for mesorregion in json.loads(requests_result.text)]

        # br_info = pd.DataFrame(uf)
        # br_info['customer_regiao'] = br_info['regiao'].apply(lambda x: x['nome'])
        # br_info.drop('regiao', axis=1, inplace=True)
        # br_info.drop_duplicates(inplace=True)

        # # El lugar más al norte de Brasil está en 5 deg 16′ 27.8″ N lat
        # geo_prep = olist_geolocation[olist_geolocation.geolocation_lat <= 5.27438888]
        # # El lugar más occidental está en 73 deg, 58′ 58.19″W long.
        # geo_prep = geo_prep[geo_prep.geolocation_lng >= -73.98283055]
        # # EL lugar mas al sur esta en 33 deg, 45′ 04.21″ S lat
        # geo_prep = geo_prep[geo_prep.geolocation_lat >= -33.75116944]
        # # Su lugar más oriental esta 34 deg, 47′ 35.33″ W long
        # geo_prep = geo_prep[geo_prep.geolocation_lng <=  -34.79314722]
        # geo_group = geo_prep.groupby(by='geolocation_zip_code_prefix', as_index=False).min()

        # # Merging all the informations
        # df_orders_items = df_orders_items.merge(br_info, how='left', left_on='customer_state', right_on='sigla')
        # df_orders_items = df_orders_items.merge(geo_group, how='left', left_on='customer_zip_code_prefix', 
        #                                         right_on='geolocation_zip_code_prefix')

        # all_data = pd.merge(all_data, df_orders_items[['customer_id','customer_regiao']], on ='customer_id')
        
        return all_data

    def generate_rfm(self, all_data):
        # ## 4.1 Recency, Recency y Monetary value
        # 
        # Deberemos crear estas variables para cada cliente antes de realizar la segmentacion.

        df = pd.DataFrame(all_data['customer_id'].unique())
        df.columns = ['customer_id']

        # ### 4.1.2 Recency
        # Para calcular recency, vamos utilizar la delivered order_purchase_timestamp

        recency_df = all_data.groupby('customer_id').order_purchase_timestamp.max().reset_index()
        recency_df.columns = ['customer_id', 'max_purchase_date']
        recency_df['recency'] = (recency_df['max_purchase_date'].max() - recency_df['max_purchase_date']).dt.days
        df = pd.merge(df, recency_df[['customer_id','recency', 'max_purchase_date']], on='customer_id')
        df.head()

        # ### 4.1.2 Frequency
        # Para encontrar frequency, es necesario encontrar el numero total de pedidos para cada cliente.

        frequency_df = all_data.groupby('customer_id').order_item_id.count().reset_index()
        frequency_df.columns = ['customer_id','frequency']
        df = pd.merge(df, frequency_df, on='customer_id')
        df.head()

        # ### 4.1.3 Monetary Value
        # Para encontrar el monetary value sumamos todos los pedidos realizados por cada cliente.

        monetary_df = all_data['payment_value'].groupby(all_data['customer_id']).sum().reset_index()
        monetary_df.columns = ['customer_id','monetary_value']
        df = pd.merge(df, monetary_df, on='customer_id')
        
        return df

    def normalize_df(self, df_rm):
        # ### 4.2.1 Tratamiento de outliers
        # Para encontrar el monetary value sumamos todos los pedidos realizados por cada cliente.
        df_rm = df_rm[['recency', 'monetary_value']]
    
        # Funcion para el tratamiento de los outliers
        def outlier_treatment(df):
            numerical_columns = df.select_dtypes(include=['int32','int64', 'float64']).columns
            for column in numerical_columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3-Q1
                df.loc[ df[column] > (1.5*IQR), column ] = ( 1.5*IQR )
                df.loc[ df[column] < (-1.5*IQR), column] = ( -1.5*IQR )
    
        outlier_treatment(df_rm)

        # ### 4.2.2 Normalizacion
        df_rm_normal = df_rm.copy()

        scaler = StandardScaler()
        scaler.fit(df_rm_normal)
        df_rm_normal = scaler.transform(df_rm_normal)
        df_rm_normal = pd.DataFrame(df_rm_normal, index=df_rm.index)
        df_rm_normal.columns=['recency', 'monetary_value']

        return df_rm_normal