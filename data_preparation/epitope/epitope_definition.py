import os

import pandas as pd
from Bio.PDB import PDBParser
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# get coordinate of sites
parser = PDBParser()
structure = parser.get_structure('7u4e', '7u4e.pdb')
alpha_C_coordinates = {}

for model in structure:
    for chain in model:
        if chain.id == 'A':
            for residue in chain:
                residue_number = residue.id[1]
                for atom in residue:
                    if atom.name == 'CA':
                        coordinates = list(atom.get_coord())
                        alpha_C_coordinates[residue_number] = coordinates

# epitope sites
data = pd.read_csv('./predictions_7u4e.csv')
top_a = list(data[(data['Chain'] == 'A') & (data['Binding site probability'] > 0.1)]['Residue Index'])
top_b = list(data[(data['Chain'] == 'B') & (data['Binding site probability'] > 0.1)]['Residue Index'])
top_c = list(data[(data['Chain'] == 'C') & (data['Binding site probability'] > 0.1)]['Residue Index'])
top_d = list(data[(data['Chain'] == 'D') & (data['Binding site probability'] > 0.1)]['Residue Index'])
epitope_sites = list(set(top_a + top_b + top_c + top_d))

epitope_sites_coord = {}
for i in epitope_sites:
    epitope_sites_coord[i] = alpha_C_coordinates[i]

# KMeans
os.environ["OMP_NUM_THREADS"] = '1'
evaluation_scores = []
for k in range(3, 9):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(list(epitope_sites_coord.values()))
    labels = kmeans.labels_
    score = silhouette_score(list(epitope_sites_coord.values()), labels)
    evaluation_scores.append((k, score))

best_k, best_score = max(evaluation_scores, key=lambda x: x[1])
print("Best k value:", best_k, "with Silhouette Score:", best_score)

# result
n_clusters = best_k
best_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
best_kmeans.fit(list(epitope_sites_coord.values()))
labels = best_kmeans.labels_
clusters = {}
for sample, label in zip(epitope_sites_coord.keys(), labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(sample)
epitopes_index = {}
for cluster, samples in clusters.items():
    print("Epitope", cluster, ":", samples)
    epitopes_index[cluster] = samples

# drop outliers
drop_sites = [358, 385, 384, 390]
for cluster, samples in clusters.items():
    print("\'Epitope_:'", samples)
    for i in drop_sites:
        if i in samples:
            samples.remove(i)
    epitopes_index[cluster] = samples
