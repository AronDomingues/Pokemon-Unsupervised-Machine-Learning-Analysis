#%% Importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Database

db_pokemon = pd.read_csv('pokemon.csv')

#%% Keeping it old school meaning 1st generation only!

db_pokemon_1stgen = db_pokemon[db_pokemon['generation'] == 1]
print(db_pokemon_1stgen.info())
descritive_db_pokemon = db_pokemon_1stgen.describe()

#%% 3d chart of observations based on initial stats = 'HP', 'Attack', 'Defense'
fig = px.scatter_3d(db_pokemon_1stgen, 
                    x='hp', 
                    y='attack', 
                    z='defense',
                    text=db_pokemon.name)
fig.show()

#%%  selecting variables for clustering

db_pokemon_1stgen_stats = db_pokemon_1stgen[['hp','attack','defense','sp_attack','sp_defense']]

#%% Agglomerative Hierarchical Cluster: Euclidian + single linkage

euclidian = pdist(db_pokemon_1stgen_stats, metric='euclidean')

#%% Dendogram for single method

plt.figure(figsize=(16,8))
dend_sing = sch.linkage(db_pokemon_1stgen_stats, method = 'single', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 4.5, labels = list(db_pokemon_1stgen.name))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pokemon', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Dendogram for complete method

plt.figure(figsize=(16,8))
dend_comp = sch.linkage(db_pokemon_1stgen_stats, method = 'complete', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_comp, color_threshold = 4.5, labels = list(db_pokemon_1stgen.name))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pokemon', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Dendogram for average method

plt.figure(figsize=(16,8))
dend_avg = sch.linkage(db_pokemon_1stgen_stats, method = 'average', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_avg, color_threshold = 4.5, labels = list(db_pokemon_1stgen.name))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pokemon', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Elbow - suggestion for the quantity of clusters

elbow = []
K = range(1,40) # stop point should be a manual input 
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(db_pokemon_1stgen_stats)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,40))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#According to the analysis we decided to keep 6 clusters

#%% Generating variable for the cluster in the database
cluster_complete = AgglomerativeClustering(n_clusters = 6, metric = 'euclidean', linkage = 'complete')
indica_cluster_complete = cluster_complete.fit_predict(db_pokemon_1stgen_stats)
db_pokemon_1stgen['cluster_complete'] = indica_cluster_complete
db_pokemon_1stgen['cluster_complete'] = db_pokemon_1stgen['cluster_complete'].astype('category')

#%% Clustering by the method K-means as well

kmeans = KMeans(n_clusters=6, init='random', random_state=100).fit(db_pokemon_1stgen_stats)

kmeans_clusters = kmeans.labels_
db_pokemon_1stgen['cluster_kmeans'] = kmeans_clusters
db_pokemon_1stgen['cluster_kmeans'] = db_pokemon_1stgen['cluster_kmeans'].astype('category')

#%% Plotting the clusters using the Agglomerative Clustering method
fig_clustering_agglomerative = px.scatter_3d(db_pokemon_1stgen, 
                    x='hp', 
                    y='attack', 
                    z='defense',
                    color='cluster_complete',
                    text=db_pokemon_1stgen.name)
fig_clustering_agglomerative.show()

#%% Plotting the clusters using the k-means method
fig_clustering_kmeans = px.scatter_3d(db_pokemon_1stgen, 
                    x='hp', 
                    y='attack', 
                    z='defense',
                    color='cluster_kmeans',
                    text=db_pokemon_1stgen.name)
fig_clustering_kmeans.show()
