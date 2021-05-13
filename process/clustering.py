from river import cluster
from river import metrics
import pandas as pd
import json


class Clustering():
    def __init__(self):
        self.incremental_kmeans = cluster.KMeans(
            n_clusters=5, halflife=0.4, sigma=3, seed=0)
        self.metric_ssw = metrics.cluster.SSW()
        self.metric_cohesion = metrics.cluster.Cohesion()
        self.metric_separation = metrics.cluster.Separation()
        self.metric_ssb = metrics.cluster.SSB()
        self.metric_bic = metrics.cluster.BIC()
        self.metric_silhouette = metrics.cluster.Silhouette()
        self.metric_xieBeni = metrics.cluster.XieBeni()

    def generate_cluster(self, df):
        # Incremental k-means
        for row in df.to_dict('records'):
            self.incremental_kmeans.learn_one(row)
            prediction = self.incremental_kmeans.predict_one(row)
            self.metric_ssw.update(
                row, prediction, self.incremental_kmeans.centers)
            self.metric_cohesion.update(
                row, prediction, self.incremental_kmeans.centers)
            self.metric_separation.update(
                row, prediction, self.incremental_kmeans.centers)
            self.metric_ssb.update(
                row, prediction, self.incremental_kmeans.centers)
            self.metric_bic.update(
                row, prediction, self.incremental_kmeans.centers)
            self.metric_silhouette.update(
                row, prediction, self.incremental_kmeans.centers)
            self.metric_xieBeni.update(
                row, prediction, self.incremental_kmeans.centers)
       
        json_centers = pd.DataFrame(
            self.incremental_kmeans.centers).transpose().to_json(orient="records")
      
        json_centers = json.loads(json_centers)
       
        metrics = {'metric_ssw_value': self.metric_ssw.get(
        ),  # 'metric_ssw_better': self.metric_ssw.bigger_is_better,
            'metric_cohesion_value': self.metric_cohesion.get(
        ),  # 'metric_cohesion_better': self.metric_cohesion.bigger_is_better,
            'metric_separation_value': self.metric_separation.get(
        ),  # 'metric_separation_better': self.metric_separation.bigger_is_better,
            'metric_ssb_value': self.metric_ssb.get(
        ),  # 'metric_ssb_better': self.metric_ssb.bigger_is_better,
            'metric_bic_value': self.metric_bic.get(
        ),  # 'metric_bic_better': self.metric_bic.bigger_is_better,
            'metric_silhouette_value': self.metric_silhouette.get(
        ),  # 'metric_silhouette_better': self.metric_silhouette.bigger_is_better,
            #'metric_xieBeni_value': self.metric_xieBeni.get(
        #),  # 'metric_xieBeni_better': self.metric_xieBeni.bigger_is_better
        }

        clusters = json_centers

        return [metrics, clusters]
