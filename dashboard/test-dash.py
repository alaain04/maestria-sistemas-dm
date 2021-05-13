import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

import dash_table
import pandas as pd


olist_orders = pd.read_csv('../dataset/olist_orders_dataset.csv')
olist_products = pd.read_csv('../dataset/olist_products_dataset.csv')
olist_items = pd.read_csv('../dataset/olist_order_items_dataset.csv')
olist_customers = pd.read_csv(
    '../dataset/olist_customers_dataset.csv')
olist_payments = pd.read_csv(
    '../dataset/olist_order_payments_dataset.csv')
olist_sellers = pd.read_csv('../dataset/olist_sellers_dataset.csv')
olist_geolocation = pd.read_csv(
    '../dataset/olist_geolocation_dataset.csv')
olist_reviews = pd.read_csv(
    '../dataset/olist_order_reviews_dataset.csv')
olist_product_category_name = pd.read_csv(
    '../dataset/product_category_name_translation.csv')

# Mergeamos los df anteriores en uno solo
df = olist_orders.merge(olist_items, on='order_id', how='left')
df = df.merge(olist_payments, on='order_id', how='inner')
df = df.merge(olist_reviews, on='order_id', how='inner')
df = df.merge(olist_products, on='product_id', how='inner')
df = df.merge(olist_customers, on='customer_id', how='inner')
df = df.merge(olist_sellers, on='seller_id', how='inner')
df = df.merge(olist_product_category_name,
              on='product_category_name', how='inner')

df = df[:100]

top_20_city_shopping = df['order_item_id'].groupby(
    df['customer_city']).sum().sort_values(ascending=False)[:20]

# # Visualizacion
# fig = plt.figure(figsize=(16, 9))
# sns.barplot(y=top_20_city_shopping.index,
#             x=top_20_city_shopping.values, palette=sns.color_palette())
# plt.title('Top 20 ciudades que más compran', fontsize=20)
# plt.xlabel('Cantidad de productos', fontsize=17)
# plt.ylabel('Ciudad', fontsize=17)


app = dash.Dash(__name__)


top_20_city_selling = df['order_item_id'].groupby(
    df['seller_state']).sum().sort_values(ascending=False)[:200]

selling_states_fig = go.Figure(data=[go.Bar(x=list(top_20_city_selling.values),
                             y=list(top_20_city_selling.index),
                             orientation='h',
                             )],
                layout=go.Layout(title='Top 20 ciudades que más compran', xaxis=dict(
                                 title='Cantidad'), yaxis=dict(title='Estado'), hovermode='closest'))

# shipping_amount_fig = go.Figure(data=[go.Bar(x=list(top_20_city_selling.values),
#                              y=list(top_20_city_selling.index),
#                              orientation='h',
#                              )],
#                 layout=go.Layout(title='Top 20 ciudades que más compran', xaxis=dict(
#                                  title='Cantidad'), yaxis=dict(title='Estado'), hovermode='closest'))

# plt.figure(figsize=(10, 8))
# ax = (all_data.groupby('customer_regiao')['freight_value'].mean().round(
#     2)).plot(kind='bar', width=0.8, color=colores, edgecolor='white')
# for p in ax.patches:
#     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1))

# plt.xlabel('')
# plt.ylabel('R$')
# plt.title('Coste medio de envío por región')
# plt.xticks(fontweight='bold', rotation='horizontal')

# plt.show()

app.layout = html.Div([
    # [dash_table.DataTable(
    #     id='table',
    #     columns=[{"name": i, "id": i} for i in df.columns],
    #     data=df.to_dict('records'),
    # ),
    dcc.Graph(
        id='top_20_cities_fig',
        figure=selling_states_fig
    )
]
)


app.run_server(host='0.0.0.0', port=8080, debug=True)
