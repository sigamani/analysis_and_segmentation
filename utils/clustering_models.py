import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA


class PrincipleComponentsAnalysis:

    def __init__(self, input_data, comp_n):
        self.data = input_data
        self.n_component = comp_n

    def reduce(self):
        pca_in = self.data.copy()
        pca_n = PCA(n_components=self.n_component,
                    copy=True, whiten=False, svd_solver='auto', tol=0.0,
                    iterated_power='auto', random_state=0)
        model = pca_n.fit(pca_in)
        trans_pca = pca_n.transform(pca_in)
        df_pca = pd.DataFrame(trans_pca)
        variance_ratio = model.explained_variance_ratio_
        print(f'PCA components: {self.n_component}, describes: {round(sum(variance_ratio), 2)}')
        return df_pca


class KMeansClustering:

    def __init__(self, input_data, n_clusters):
        self.data = input_data
        self.n_clusters = n_clusters

    def train(self):
        model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=4, max_iter=600, tol=0.0001,
                       precompute_distances='auto', verbose=0, random_state=1, copy_x=True, n_jobs=1, algorithm='auto')
        model.fit(self.data)
        return model

    def infer_and_test(self, model):
        model.fit(self.data)
        labels = model.labels_
        ch_score = calinski_harabasz_score(self.data, model.labels_)
        s_score = silhouette_score(self.data, model.labels_)
        print(f"KMeans: {self.n_clusters} clusters, "
              f"Calinski-Harabaz_score: {round(ch_score, 2)}, " 
              f"Silhouette score: {round(s_score, 2)}")
        return labels, ch_score, s_score


class MixtureOfGaussianClustering:

    def __init__(self, input_data, n_clusters):
        self.data = input_data
        self.n_clusters = n_clusters

    def train(self):
        model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=4, max_iter=600, tol=0.0001,
                       precompute_distances='auto', verbose=0, random_state=1, copy_x=True, n_jobs=1, algorithm='auto')
        model.fit(self.data)
        return model

    def infer_and_test(self, model):
        model.fit(self.data)
        labels = model.labels_
        ch_score = calinski_harabasz_score(self.data, model.labels_)
        s_score = silhouette_score(self.data, model.labels_)
        print(f"KMeans: {self.n_clusters} clusters, "
              f"Calinski-Harabaz_score: {round(ch_score, 2)}, " 
              f"Silhouette score: {round(s_score, 2)}")
        return labels, ch_score, s_score


def pca_3d(df):
    pca = PCA(n_components=3,
              copy=True,
              whiten=False,
              svd_solver='auto',
              tol=0.0,
              iterated_power='auto',
              random_state=0)
    X_pca = pca.fit(df)
    trans_pca = X_pca.transform(df)
    pca3_df = pd.DataFrame(trans_pca)
    pca3_df.columns = ['x', 'y', 'z']
    print('explained_variance_ratio', X_pca.explained_variance_ratio_)
    return pca3_df
