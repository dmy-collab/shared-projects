import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
pd.set_option('display.max_columns', None)


# Reading the file:
dataset = pd.read_csv('SARS_expression.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.isna().sum())

# Separating values for infected samples:
new_data = dataset.iloc[:, 5:]
print(new_data.head())

# Determining the optimal value of K using the elbow method:
wcss = []   # within cluster sum of squares
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=150)
    kmeans.fit(new_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss, color='k')
plt.title("The Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()      # from the graph, 18 - is the optimal value
print(kmeans.labels_)

# Assigning the cluster numbers retrieved from the step above to the ID_REF:
cluster_genes =\
    pd.concat([dataset.iloc[:, 0], pd.DataFrame(kmeans.labels_)], axis=1)
cluster_genes.columns = ['ID_REF', 'Groups']
print(cluster_genes.head())

# Grouping genes into respective clusters
cluster = cluster_genes.groupby('Groups')
print(cluster.describe())       # cluster 16 has the highest N of genes

# retrieving particular cluster:
cluster = {k: v for k, v in cluster_genes.groupby('Groups')}
print('\nCluster with the highest number of genes:\n')
print(cluster[16])
