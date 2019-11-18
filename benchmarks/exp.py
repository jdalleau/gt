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
from networkx.generators import community, erdos_renyi_graph


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
        print("Run number {0}".format(i))
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
    print("Structure : {0}, {1}".format(np.mean(nmis_structure), np.std(nmis_structure)))
    print("Attributes : {0}, {1}".format(np.mean(nmis_attributes), np.std(nmis_attributes)))
    print("Both : {0}, {1}".format(np.mean(nmis_both), np.std(nmis_both)))

    return(nmis_structure, nmis_attributes, nmis_both)

def test_clustering_structure(n_runs=20):
    nmis_gt = []
    nmis_mcl = []
    nmis_louvain = []
    for i in range(n_runs):
        print("Run number {0}".format(i))
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
    print("GT : {0}, {1}".format(np.mean(nmis_gt), np.std(nmis_gt)))
    print("MCL : {0}, {1}".format(np.mean(nmis_mcl), np.std(nmis_mcl)))
    print("Louvain : {0}, {1}".format(np.mean(nmis_louvain), np.std(nmis_louvain)))
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

print("--Football--")
G = nx.read_gml("../data/football.gml")
true = []
adj = nx.adjacency_matrix(G).todense()

with open("../data/football.gml") as f:
    for line in f:
        values = line.split(" ")
        # print(values)
        if(len(values) >= 5):
            if(values[4] == "value"):
                    true.append(values[5])
encoder = LabelEncoder()
true = encoder.fit_transform(true)
model_hac = hac(n_clusters=len(set(true)), affinity="precomputed",linkage="average")
model_kmeans = KMeans(n_clusters=len(set(true)))
tsne = TSNE(n_components=2, metric='precomputed')
np.savetxt('file.csv', adj, delimiter='\t')
test_clustering_structure()


####################

####################

print("--Pokbooks---")
G = nx.read_gml("../data/polbooks.gml")
true = []
adj = nx.adjacency_matrix(G).todense()

with open("../data/polbooks.gml") as f:
    for line in f:
        values = line.split(" ")
        # print(values)
        if(len(values) >= 5):
            if(values[4] == "value"):
                true.append(values[5])
encoder = LabelEncoder()
true = encoder.fit_transform(true)

tsne = TSNE(n_components=2, metric='precomputed')
np.savetxt('file.csv', adj, delimiter='\t')
test_clustering_structure()

####################

####################
print("---Email---")
G = nx.read_edgelist("../data/email-Eu-core.txt")
print(nx.info(G))
true = np.loadtxt("../data/email-Eu-core-department-labels.txt", dtype=np.int16)[:,1]
adj = nx.adjacency_matrix(G).todense()
encoder = LabelEncoder()
true = encoder.fit_transform(true)
np.savetxt('file.csv', adj, delimiter='\t')
test_clustering_structure()

####################

####################
print("---SBM---")
probs = [[0.95, 0.55, 0.02],
         [0.55, 0.95, 0.55],
         [0.02, 0.55, 0.95]]
size = 300
sizes = [size, size, size]

G = community.stochastic_block_model(sizes, probs)
true = [[i]*size for i in range(3)]
true = sum(true, [])
adj = nx.adjacency_matrix(G).todense()
np.savetxt('file.csv', adj, delimiter='\t')
test_clustering_structure()

####################

####################
print("---Polblogs---")
G = nx.read_gml("../data/polblogs.gml")

G = G.to_undirected()
print(nx.info(G))
true = []

attributes = []
for node, data in G.nodes(data=True):
    true.append(int(data["value"]))
    attributes.append(data["source"])
attributes = attributes[:len(list(G.nodes))]
attributes_clean = []
for attribute in attributes:
    attributes_clean.append(attribute.strip().split(","))
set_attributes = set()
attributes = []
for attribute in attributes_clean:
    for i in attribute:
        if(i[0]=='"'):
            i = i[1:]
        if(i[-1]=='"'):
            i = i[:-1]
        set_attributes.add(i)
        attributes.append(i)
encoder = LabelEncoder()
encoder.fit(list(set_attributes))
attributes_encoded = encoder.transform(list(set_attributes))
X = np.zeros((len(list(G.nodes)), len(set_attributes)))

for i in range(len(list(G.nodes))):
    for local_attributes in attributes_clean[i]:
        if(local_attributes[0]=='"'):
            local_attributes = local_attributes[1:]
        if(local_attributes[-1]=='"'):
            local_attributes = local_attributes[:-1]
        node_attributes = encoder.transform([local_attributes])
        X[i][node_attributes[0]] = 1
print(X.shape)

adj = nx.adjacency_matrix(G).todense()

np.savetxt('file.csv', adj, delimiter='\t')
np.savetxt('file_attributes.csv', X, delimiter='\t')
nmis_structure, nmis_attributes, nmis_both = test_clustering()

####################

####################

print("---Parliament---")
A = sio.mmread('../data/parliament/A.mtx')
X = sio.mmread("../data/parliament/X.mtx")

true = np.load("../data/parliament/z.npy")
adj = A.todense()
G = nx.Graph(adj)
adj = nx.adjacency_matrix(G).todense()
X = X.todense()
n_clusters = len(set(true))
X = pd.DataFrame(X)

np.savetxt('file.csv', adj, delimiter='\t')
np.savetxt('file_attributes.csv', X, delimiter='\t')
nmis_structure, nmis_attributes, nmis_both = test_clustering()


####################

####################


print("---HVR---")
A = sio.mmread('../data/hvr/A.mtx')
X = sio.mmread("../data/hvr/X.mtx")

true = np.load("../data/hvr/z.npy")
adj = A.todense()
G = nx.Graph(adj)
coltypes = [1]*X.todense().shape[1]
adj = nx.adjacency_matrix(G).todense()
X = X.todense()
n_clusters = len(set(true))
X = pd.DataFrame(X)

np.savetxt('file.csv', adj, delimiter='\t', fmt='%u')
np.savetxt('file_attributes.csv', X, delimiter='\t', fmt='%f')
nmis_structure, nmis_attributes, nmis_both = test_clustering()

####################

####################

print("---Lawyers---")
A = sio.mmread('../data/lawyers/A.mtx')
X = sio.mmread("../data/lawyers/X.mtx")

true = np.load("../data/lawyers/z.npy")
adj = A.todense()
G = nx.Graph(adj)
coltypes = [1]*X.todense().shape[1]
adj = nx.adjacency_matrix(G).todense()
X = X.todense()
n_clusters = len(set(true))
X = pd.DataFrame(X)

np.savetxt('file.csv', adj, delimiter='\t')
np.savetxt('file_attributes.csv', X, delimiter='\t')
nmis_structure, nmis_attributes, nmis_both = test_clustering()

####################

####################

print("---WebKB---")
le = LabelEncoder()
G = nx.read_edgelist("../data/WebKB/edgelist")
G = G.to_undirected()
X = pd.read_csv("../data/WebKB/attributes", header=None, sep="\t")

true = [0]*195+[1]*187+[2]*230+[3]*265
X = X[X.columns[:-1]]
indices = [X.loc[X[0]==i].index.values.astype(int)[0] for i in G.nodes]
X = X[X.columns[1:]]
X = np.array(X, dtype=np.float)
true = np.array(true)
n_clusters = 4

coltypes = [1]*X.shape[1]
adj = nx.adjacency_matrix(G).todense()


np.savetxt('file.csv', adj, delimiter='\t')
np.savetxt('file_attributes.csv', X, delimiter='\t')
nmis_structure, nmis_attributes, nmis_both = test_clustering()
