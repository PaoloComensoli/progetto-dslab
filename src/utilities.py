import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd


def associate_sector_to_category(cluster_predicted, n_clusters):
    """
    associates each time series to its own cluster"""
    clusters = {}
    category = [element for element in set(df['settore'])]
    for n_cluster in range(n_clusters):
        clusters[n_cluster] = []
    for predicted_cluster in pd.Series(cluster_predicted).index:
        for cluster in clusters.keys():
          if pd.Series(cluster_predicted)[predicted_cluster] == cluster:
            clusters[cluster].append(category[predicted_cluster])
          else:
            pass
    return clusters


def plot_clusters(n_clusters, fitted_model):
    """
    plots the obtained clusters 
    """
    fig, ax = plt.subplots(n_clusters,1, figsize =(20,15))
    for clusters in range(n_clusters):
        ax[clusters].plot(processed_df.data,fitted_model.cluster_centers_[clusters])


def pivot_scale_and_fillna(df):
    """
    returns a pivoted version of the dataframe where the values 
    are scaled and null values substituted with zeros
    """
    scaler = sklearn.preprocessing.StandardScaler()
    df_pivoted = pd.pivot(data = df, index = 'data', columns = 'settore', values = 'totale')
    col_names = [col_name for col_name in df_pivoted.columns]
    index_ = df_pivoted.index
    df_pivoted = df_pivoted.fillna(0)
    df_pivoted = pd.DataFrame(scaler.fit_transform(df_pivoted), columns = col_names, index = index_)
    return df_pivoted


