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
from networkx.generators.random_graphs import erdos_renyi_graph as erg

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

def compute_sim_intra_inter(similarity,classes):
	distinct_classes = np.unique(classes)
	indices = []
	for current in distinct_classes:
		indices.append(np.where(classes==current)[0])
	intrac_similarities = []
	interc_similarities = []
	for i in range(len(indices)):
		cluster1_indices = indices[i]
		for j in range(len(indices)):
			cluster2_indices = indices[j]
			if(i==j):
				local_sim = [similarity[i][j] for i in cluster1_indices for j in cluster2_indices if (i != j) ]
				intrac_similarities.extend(local_sim)
			else:
				local_sim = [similarity[i][j]  for i in cluster1_indices for j in cluster2_indices if (i != j)]
				interc_similarities.extend(local_sim)
	return((intrac_similarities, interc_similarities))

from sklearn.cluster.hierarchical import AgglomerativeClustering as hac

def get_dist(dist, true):
    sims = compute_sim_intra_inter(dist, true)
    intrac = sims[0]
    interc = sims[1]
    intrac_mean = np.mean(intrac)
    interc_mean = np.mean(interc)
    return(abs(interc_mean-intrac_mean))

def test_clustering_structure(n_runs=20):
    dist_gt = []

    for i in range(n_runs):

        ensemble_density_huge("file.csv", "\\t")
        dist_dense = pd.read_csv("./matrix.csv", delimiter="\t", header=None).values
        dist_dense = dist_dense[:,:-1]
        sims = compute_sim_intra_inter(dist_dense, true)
        intrac = sims[0]
        interc = sims[1]
        intrac_mean = np.mean(intrac)
        interc_mean = np.mean(interc)
        dist_gt.append(abs(interc_mean-intrac_mean))

    return(dist_gt)

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

dist = []
####################

####################
print("--Random graph--")
G = erg(300, 0.2)

adj = nx.adjacency_matrix(G).todense()
np.savetxt('file.csv', adj, delimiter='\t')

true_random = [i for i in range(len(G.nodes))]
dists_gt = ensemble_density_huge("file.csv", "\\t")
dist_dense = pd.read_csv("./matrix.csv", delimiter="\t", header=None).values
dist_dense = dist_dense[:,:-1]
dist.append(dist_dense)

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
true_sbm = true
adj = nx.adjacency_matrix(G).todense()
np.savetxt('file.csv', adj, delimiter='\t')
dists_gt = test_clustering_structure()
print("GT : {0}, {1}".format(np.mean(dists_gt), np.std(dists_gt)))
dists_gt = ensemble_density_huge("file.csv", "\\t")
dist_dense = pd.read_csv("./matrix.csv", delimiter="\t", header=None).values
dist_dense = dist_dense[:,:-1]
dist.append(dist_dense)

####################

####################

print("--Football--")
G = nx.read_gml("../data/football.gml")
true = []
adj = nx.adjacency_matrix(G).todense()

with open("../data/football.gml") as f:
    for line in f:
        values = line.split(" ")
        if(len(values) >= 5):
            if(values[4] == "value"):
                    true.append(values[5])
encoder = LabelEncoder()
true = encoder.fit_transform(true)
true_football = true
model_hac = hac(n_clusters=len(set(true)), affinity="precomputed",linkage="average")
model_kmeans = KMeans(n_clusters=len(set(true)))
tsne = TSNE(n_components=2, metric='precomputed')
np.savetxt('file.csv', adj, delimiter='\t')
test_clustering_structure()
dists_gt = test_clustering_structure()
print("GT : {0}, {1}".format(np.mean(dists_gt), np.std(dists_gt)))
dists_gt = ensemble_density_huge("file.csv", "\\t")
dist_dense = pd.read_csv("./matrix.csv", delimiter="\t", header=None).values
dist_dense = dist_dense[:,:-1]
dist.append(dist_dense)



tsne = TSNE(n_components=2, metric='precomputed')
projections = []
for distances in dist:
    projections.append(tsne.fit_transform(distances))

results_random, results_sbm, results_football = projections
plt.switch_backend('agg')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.set_title("Random graph")
ax2.set_title("SBM")
ax3.set_title("Football")
for i in range(len(results_random)):
    ax1.scatter(results_random[i, 0], results_random[i, 1], c=colors[true_random[i]-1])
for i in range(len(results_sbm)):
    ax2.scatter(results_sbm[i, 0], results_sbm[i, 1], c=colors[true_sbm[i]-1])
for i in range(len(results_football)):
    ax3.scatter(results_football[i, 0], results_football[i, 1], c=colors[true_football[i]-1])

f.set_size_inches((15,7))
plt.savefig('./dist.svg', format='svg')