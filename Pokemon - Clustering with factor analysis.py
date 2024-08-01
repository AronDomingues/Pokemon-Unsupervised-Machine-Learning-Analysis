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
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import sympy as sy

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
                    text=db_pokemon_1stgen.name)
fig.show()

#%%  selecting variables for PCA

db_pokemon_1stgen_stats = db_pokemon_1stgen[['hp','attack','defense','sp_attack','sp_defense']]

#%% Pearson correlation of the variables

pearson_correlation_stats=pg.rcorr(db_pokemon_1stgen_stats, method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Bartlett's test of sphericity

bartlett, p_value = calculate_bartlett_sphericity(db_pokemon_1stgen_stats)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-value: {round(p_value, 4)}')

#p-value grants we can use this statistical approach

#%% Defining PCA (1st with all available factors)

fa = FactorAnalyzer(n_factors=5, method='principal', rotation=None).fit(db_pokemon_1stgen_stats)

#%% Obtaining Eigenvalues, variances and cumulative variances 

eigenvalues_factors= fa.get_factor_variance()

table_eigen = pd.DataFrame(eigenvalues_factors)
table_eigen.columns = [f"Factor {i+1}" for i, v in enumerate(table_eigen.columns)]
table_eigen.index = ['Eigenvalue','Variance', 'Cumulative Variance']
table_eigen = table_eigen.T

print(table_eigen)

# It's possible to see that 3 factors result in 83% of the cumulative variance.

#%% Obtaining factorial loadings

factorial_loadings = fa.loadings_

table_loads = pd.DataFrame(factorial_loadings)
table_loads.columns = [f"Factor {i+1}" for i, v in enumerate(table_loads.columns)]
table_loads.index = db_pokemon_1stgen_stats.columns

print(table_loads)

#%% Analyzing factorial loads of each factor 

table_loads_graph = table_loads.reset_index()
table_loads_graph = table_loads_graph.melt(id_vars='index')

sns.barplot(data=table_loads_graph, x='variable', y='value', hue='index', palette='bright')
plt.legend(title='Variables', bbox_to_anchor=(1,1), fontsize = '6')
plt.title('Factorial loads', fontsize='12')
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.show()

#%% It's possible to see that 3 factors result in 83% of the cumulative variance. We'll use only those 3 factors as the variables for the clustering of pokemons.

fa = FactorAnalyzer(n_factors=3, method='principal', rotation=None).fit(db_pokemon_1stgen_stats)

#%% Obtaining Eigenvalues, variances and cumulative variances for new PCA 

eigenvalues_factors= fa.get_factor_variance()

table_eigen = pd.DataFrame(eigenvalues_factors)
table_eigen.columns = [f"Factor {i+1}" for i, v in enumerate(table_eigen.columns)]
table_eigen.index = ['Eigenvalue','Variance', 'Cumulative Variance']
table_eigen = table_eigen.T

print(table_eigen)

# It's possible to see that 3 factors result in 83% of the cumulative variance.

#%% Obtaining factorial loadings for new PCA

factorial_loadings = fa.loadings_

table_loads = pd.DataFrame(factorial_loadings)
table_loads.columns = [f"Factor {i+1}" for i, v in enumerate(table_loads.columns)]
table_loads.index = db_pokemon_1stgen_stats.columns

print(table_loads)

#%% Extracting factors to the observations on the database

factors = pd.DataFrame(fa.transform(db_pokemon_1stgen_stats))
factors.columns =  [f"Factor {i+1}" for i, v in enumerate(factors.columns)]

# Adding factors to the database

db_pokemon_1stgen = pd.concat([db_pokemon_1stgen.reset_index(drop=True),factors], axis=1)

#%% Creating a rank based on the 3 existing factors

db_pokemon_1stgen['Ranking'] = 0

for index, item in enumerate(list(table_eigen.index)):
    variance = table_eigen.loc[item]['Variance']

    db_pokemon_1stgen['Ranking'] = db_pokemon_1stgen['Ranking'] + db_pokemon_1stgen[table_eigen.index[index]]*variance

#%%  selecting variables for clustering

db_pokemon_1stgen_factor_stats = db_pokemon_1stgen[['Factor 1','Factor 2','Factor 3']]

#%% Agglomerative Hierarchical Cluster: Euclidian + single linkage

euclidian = pdist(db_pokemon_1stgen_factor_stats, metric='euclidean')

#%% Dendogram for single method

plt.figure(figsize=(16,8))
dend_sing = sch.linkage(db_pokemon_1stgen_factor_stats, method = 'single', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 4.5, labels = list(db_pokemon_1stgen.name))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pokemon', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Dendogram for complete method

plt.figure(figsize=(16,8))
dend_comp = sch.linkage(db_pokemon_1stgen_factor_stats, method = 'complete', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_comp, color_threshold = 4.5, labels = list(db_pokemon_1stgen.name))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pokemon', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Dendogram for average method

plt.figure(figsize=(16,8))
dend_avg = sch.linkage(db_pokemon_1stgen_factor_stats, method = 'average', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_avg, color_threshold = 4.5, labels = list(db_pokemon_1stgen.name))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pokemon', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Elbow - suggestion for the quantity of clusters

elbow = []
K = range(1,40) # stop point should be a manual input 
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(db_pokemon_1stgen_factor_stats)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('# of Clusters', fontsize=16)
plt.xticks(range(1,40))
plt.ylabel('WCSS', fontsize=16)
plt.title('Elbow method', fontsize=16)
plt.show()

#According to the analysis we decided to keep 5 clusters

#%% Generating variable for the cluster in the database

cluster_complete = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'complete')
indica_cluster_complete = cluster_complete.fit_predict(db_pokemon_1stgen_factor_stats)
db_pokemon_1stgen['cluster_complete'] = indica_cluster_complete
db_pokemon_1stgen['cluster_complete'] = db_pokemon_1stgen['cluster_complete'].astype('category')

#%% Clustering by the method K-means as well

kmeans = KMeans(n_clusters=5, init='random', random_state=100).fit(db_pokemon_1stgen_factor_stats)

kmeans_clusters = kmeans.labels_
db_pokemon_1stgen['cluster_kmeans'] = kmeans_clusters
db_pokemon_1stgen['cluster_kmeans'] = db_pokemon_1stgen['cluster_kmeans'].astype('category')

#%% Plotting the clusters using the Agglomerative Clustering method
fig_clustering_agglomerative = px.scatter_3d(db_pokemon_1stgen, 
                    x='Factor 1', 
                    y='Factor 2', 
                    z='Factor 3',
                    color='cluster_complete',
                    text=db_pokemon_1stgen.name)
fig_clustering_agglomerative.show()

#%% Plotting the clusters using the k-means method
fig_clustering_kmeans = px.scatter_3d(db_pokemon_1stgen, 
                    x='Factor 1', 
                    y='Factor 2', 
                    z='Factor 3',
                    color='cluster_kmeans',
                    text=db_pokemon_1stgen.name)
fig_clustering_kmeans.show()

#%% Exporting file 

filename = 'Pokemon-clustering with factor analysis.csv'

db_pokemon_1stgen.to_csv(filename, index=False,sep =";")