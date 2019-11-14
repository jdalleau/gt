import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomTreesEmbedding as rte
from sklearn.cluster.hierarchical import AgglomerativeClustering as hac
import math
import warnings
import random
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import markov_clustering as mc
import community as louvain
import os
import json

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)


import subprocess

def ensemble_density_huge(path, sep):
    cmd = ['../main', "{0}".format(path)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    p.wait()
    return(p.returncode)

def ensemble_attributes(path, sep):
    cmd = ['../uet', "{0}".format(path)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    return(p.returncode)

def test_clustering(n_runs=20, alpha=0.5):
    nmis_both = []
    nmis_attributes = []
    nmis_structure = []
    for i in range(n_runs):
        ensemble_density_huge('file.csv', "'\t'")
        dist_dense = pd.read_csv("./matrix.csv", delimiter="\t", header=None).values
        dist_dense = dist_dense[:,:-1]



        sims_attributes = ensemble_attributes("file_attributes.csv", "\t")
        sim_attributes = pd.read_csv("./matrix_uet.csv", delimiter="\t", header=None).values
        sim_attributes = sim_attributes[:,:-1]

        dist_attributes = sim_to_dist(np.array(sim_attributes))
        dist = alpha*dist_dense + (1-alpha)*dist_attributes
        dist = dist/2
        model_kmeans = KMeans(n_clusters=len(set(true)))
        scaler = QuantileTransformer(n_quantiles=10)
        dist_scaled = scaler.fit_transform(dist)
        dist_dense_scaled = scaler.fit_transform(dist_dense)
        dist_attributes_scaled = scaler.fit_transform(dist_attributes)
        results_dense = TSNE(metric="precomputed").fit_transform(dist_dense_scaled)

        results_dense_both = TSNE(metric="precomputed").fit_transform(dist_scaled)
        results_dense_attributes = TSNE(metric="precomputed").fit_transform(dist_attributes_scaled)
        labels_dense_kmeans_both = model_kmeans.fit_predict(results_dense_both)
        labels_dense_kmeans_attributes = model_kmeans.fit_predict(results_dense_attributes)
        labels_dense_kmeans_structure = model_kmeans.fit_predict(results_dense)

        nmis_both.append(nmi(labels_dense_kmeans_both, true, average_method="arithmetic"))
        nmis_attributes.append(nmi(labels_dense_kmeans_attributes, true, average_method="arithmetic"))
        nmis_structure.append(nmi(labels_dense_kmeans_structure, true, average_method="arithmetic"))
    return(nmis_structure, nmis_attributes, nmis_both)

def test_clustering_structure(n_runs=20):
    nmis_gt = []
    nmis_mcl = []
    nmis_louvain = []
    for i in range(n_runs):

        ensemble_density_huge("file.csv", "\\t")
        dist_dense = pd.read_csv("./matrix.csv", delimiter="\t", header=None).values
        dist_dense = dist_dense[:,:-1]
        scaler = QuantileTransformer(n_quantiles=10)
        dist_dense_scaled = scaler.fit_transform(dist_dense)
        results_dense = TSNE(metric="precomputed").fit_transform(dist_dense_scaled)
        model_kmeans = KMeans(n_clusters=len(set(true)))
        labels_dense_kmeans = model_kmeans.fit_predict(results_dense)
        clusters_mcl = [0 for i in range(len(adj))]
        result_mcl = mc.run_mcl(adj)           # run MCL with default parameters
        clusters = mc.get_clusters(result_mcl)    # get clusters
        i = 0
        for cluster in clusters:
            for j in cluster:
                clusters_mcl[j] = i
            i+=1

        partition = louvain.best_partition(G)
        labels_spectral = [v for k,v in partition.items()]

        nmis_gt.append(nmi(labels_dense_kmeans, true, average_method="arithmetic"))
        nmis_mcl.append(nmi(clusters_mcl, true, average_method="arithmetic"))
        nmis_louvain.append(nmi(labels_spectral, true, average_method="arithmetic"))
    return((nmis_gt, nmis_mcl, nmis_louvain))

colors = []
r = lambda: random.randint(0,255)

for i in range(1000):
    colors.append('#%02X%02X%02X' % (r(),r(),r()))

def sim_to_dist(matrix):
    out = np.zeros((len(matrix),len(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            out[i][j] = math.sqrt(1 - matrix[i][j])
    return(out)


####################

####################



G = nx.read_edgelist("../data/git_web_ml/git_edges.csv", delimiter=",")
print(nx.info(G))
with open('../data/git_web_ml/git_features.json') as f:
  data = json.load(f)

maxFeature = 0
for key, value in data.items():
    maxValue = max(value)
    if maxValue > maxFeature:
        maxFeature = maxValue

X = np.zeros((len(G.nodes), maxFeature))
for key, features in data.items():
    for feature in features:
        X[int(key)][int(feature)-1] = 1


trueDataFrame = pd.read_csv("../data/git_web_ml/git_target.csv")
true = trueDataFrame["ml_target"].values

adj = nx.adjacency_matrix(G).todense()
G = nx.Graph(adj)
adj = nx.adjacency_matrix(G).todense()
n_clusters = len(set(true))

np.savetxt('file.csv', adj, delimiter='\t', fmt='%u')
np.savetxt('file_attributes.csv', X, delimiter='\t', fmt='%f')

start = time.time()
ensemble_density_huge('file.csv', "'\t'")



sims_attributes = ensemble_attributes("file_attributes.csv", "\t")


nmis_structure, nmis_attributes, nmis_both = test_clustering(20)
print("Structure : {0}, {1}".format(np.mean(nmis_structure), np.std(nmis_structure)))
print("Attributes : {0}, {1}".format(np.mean(nmis_attributes), np.std(nmis_attributes)))
print("Both : {0}, {1}".format(np.mean(nmis_both), np.std(nmis_both)))


# true = [0 for i in range(len(G.nodes()))]

# cluster = 0
# with open(file) as fp:  
#    line = fp.readline()
#    cnt = 1
#    while line:
#        data = line.split("\t")
#        for i in data:
#             # true[int(i)] = cluster
#             if (int(i) > 317000):
#                 print(int(i))
#        line = fp.readline()

#    cluster += 1
# # print(true)
# adj = nx.adjacency_matrix(G).todense()
# encoder = LabelEncoder()
# true = encoder.fit_transform(true)

# model_hac = hac(n_clusters=len(set(true)), affinity="precomputed",linkage="average")
# model_kmeans = KMeans(n_clusters=len(set(true)))
# tsne = TSNE(n_components=2, metric='precomputed')
# np.savetxt('file.csv', adj, delimiter='\t')
# print(adj.shape)


# A = sio.mmread('../data/hvr/A.mtx')
# X = sio.mmread("../data/hvr/X.mtx")

# true = np.load("../data/hvr/z.npy")
# adj = A.todense()
# G = nx.Graph(adj)
# coltypes = [1]*X.todense().shape[1]
# adj = nx.adjacency_matrix(G).todense()
# X = X.todense()
# n_clusters = len(set(true))
# X = pd.DataFrame(X)

# np.savetxt('file.csv', adj, delimiter='\t', fmt='%u')
# np.savetxt('file_attributes.csv', X, delimiter='\t', fmt='%f')

# start = time.time()
# ensemble_density_huge('file.csv', "'\t'")



# sims_attributes = ensemble_attributes("file_attributes.csv", "\t")


# nmis_structure, nmis_attributes, nmis_both = test_clustering(20)
# print("Structure : {0}, {1}".format(np.mean(nmis_structure), np.std(nmis_structure)))
# print("Attributes : {0}, {1}".format(np.mean(nmis_attributes), np.std(nmis_attributes)))
# print("Both : {0}, {1}".format(np.mean(nmis_both), np.std(nmis_both)))
