from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def create_clsuters(k, dataset):
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(dataset)

    # create an extra column in data for storing the cluster values
    dataset['cluster'] = kmeanModel.labels_
    dataset['cluster'].sample(n=10)

    return dataset[["user_id", "cluster"]]


# define function to calculate the clustering errors
def clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    silhouette_avg = silhouette_score(data, predictions)

    return silhouette_avg
