import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import modularity
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(nb_workers=25)

# batch mcl
path = './mcl_result/'
os.makedirs(path)
commands = []
model = 'rf'
for i in range(10, 101):
    out = 'out.I' + str(i)
    commands.append(
        f'mcl ./{model}.abc -te 6 --abc -force-connected y -I {i / 10} -o {path + out} > /dev/null 2>&1')


def run_command(command):
    try:
        print(command)
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e}")


with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(run_command, commands)


# calculate modularity

def read_cluster_results(file_path, nodes):
    clusters = {}
    community = {}
    for node in nodes:
        clusters[node] = 0
    community[0] = nodes.copy()
    with open(file_path, 'r') as file:
        cluster_num = 1
        for line in file:
            community[cluster_num] = []
            cluster_node = line.strip().split()
            for node in cluster_node:
                clusters[node] = cluster_num
                community[cluster_num].append(node)
                if node in community[0]:
                    community[0].remove(node)
            cluster_num += 1
    key_to_pop = []
    for i in community:
        if not community[i]:
            key_to_pop.append(i)
    for i in key_to_pop:
        community.pop(i)
    return clusters, community


def process_cluster_results(row, network, adjacency_frame, nodes):
    infla = row['infla']
    cluster_results_file = f'mcl_result' + f'/out.I{infla}'
    clusters, community = read_cluster_results(cluster_results_file, nodes)

    cluster_labels = [clusters[node] for node in G.nodes()]
    community_set = [set(i) for i in list(community.values())]
    community_set.remove({'strain_a', 'strain_b'})

    modularity_value = modularity(G, community_set)
    row['modularity_value'] = modularity_value
    return row


with open('./na_seq/na_seqs_mafft_normal_maffted.fasta', encoding='utf-8', errors='replace') as file:
    nodes = []
    for line in file:
        line = line.strip()
        if '>' in line:
            nodes.append(line)
adjacency_matrix = np.zeros((len(nodes), len(nodes)))
adjacency_frame = pd.DataFrame(adjacency_matrix, index=nodes, columns=nodes)

file = 'rf'
data = pd.read_csv(f'{file}.abc', sep='\t')
nodes_a = list(data['strain_a'])
nodes_b = list(data['strain_b'])
weight = list(data['result'])

for i in tqdm(range(len(nodes_a))):
    adjacency_frame.loc[nodes_a[i], nodes_b[i]] = adjacency_frame.loc[nodes_b[i], nodes_a[i]] = weight[i]

G = nx.Graph(adjacency_frame)
modularity_values = []
infla = [int(i.split('.')[1][1:]) for i in os.listdir(f'.//mcl_result/') if
         ((i.startswith('out')) and ('rag' not in i))]
infla = sorted(infla)

data = pd.DataFrame(columns=['infla', 'modularity_value'])
data['infla'] = infla
data = data.parallel_apply(process_cluster_results, nodes=nodes,
                           network=G, adjacency_frame=adjacency_frame, axis=1)
data.to_csv(f'./{file}_data.csv', index=None)
