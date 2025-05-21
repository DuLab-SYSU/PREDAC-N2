import joblib
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(nb_workers=30)

seqs = {}
with open('./na_seq/na_seqs_mafft_normal_maffted.fasta', encoding='utf-8', errors='replace') as file:
    tem_seq = tem_name = ''
    for line in file:
        line = line.strip()
        if '>' in line:
            seqs[tem_name] = tem_seq
            tem_name = line
            tem_seq = ''
        else:
            tem_seq += line
seqs[tem_name] = tem_seq
seqs.pop('')

strain_a = []
strain_b = []
seqs_name = list(seqs.keys())
for i in range(len(seqs_name)):
    for j in range(i + 1, len(seqs_name)):
        strain_a.append(seqs_name[i])
        strain_b.append(seqs_name[j])
c = {
    'strain_a': strain_a,
    'strain_b': strain_b
}
data = pd.DataFrame(c)

seq_a = [seqs[i] for i in strain_a]
seq_b = [seqs[i] for i in strain_b]

data['seq_a'] = seq_a
data['seq_b'] = seq_b


def calculate_property_difference(row, property_dict):
    seq1 = row['seq_a']
    seq2 = row['seq_b']
    differences = []
    for aa1, aa2 in zip(seq1, seq2):
        if (aa1 != aa2):
            if (aa1 in property_dict) and (aa2 in property_dict):
                property_difference = abs(float(property_dict[aa1]) - float(property_dict[aa2]))
                differences.append(property_difference)
            elif (aa1 == '-') and (aa2 in property_dict):
                tem1 = abs(max(property_dict.values()) - property_dict[aa2])
                tem2 = abs(min(property_dict.values()) - property_dict[aa2])
                differences.append(max(tem1, tem2))
            elif (aa2 == '-') and (aa1 in property_dict):
                tem1 = abs(max(property_dict.values()) - property_dict[aa1])
                tem2 = abs(min(property_dict.values()) - property_dict[aa1])
                differences.append(max(tem1, tem2))
            else:
                print(aa1, aa2)
    max_differences = sorted(differences, reverse=True)[:3]
    if max_differences:
        return sum(max_differences) / len(max_differences)
    else:
        return 0.0


def calculate_space_difference(row, space_dict, ref_seq):
    seq1 = row['seq_a']
    seq2 = row['seq_b']
    differences = []
    for aa_num in range(len(seq1)):
        if seq1[aa_num] != seq2[aa_num]:
            index = aa_num + ref_seq[aa_num].count('-')
            if index in space_dict:
                space_difference = space_dict[index]
                differences.append(space_difference)
    max_differences = sorted(differences, reverse=False)[:3]
    if max_differences:
        return sum(max_differences) / len(max_differences)
    else:
        return 0.0


def calculate_epitope_difference(row, epitope_sites):
    seq1 = row['seq_a']
    seq2 = row['seq_b']
    epitope_seq1 = [seq1[i - 1] for i in epitope_sites]
    epitope_seq2 = [seq2[i - 1] for i in epitope_sites]
    return sum([a != b for a, b in zip(epitope_seq1, epitope_seq2)])


def calculate_gly_difference(row, gly_result):
    strain1 = row['seq_a'].replace('-', '')
    strain2 = row['seq_b'].replace('-', '')
    a = set(gly_result[strain1])
    b = set(gly_result[strain2])
    diff = len(a.symmetric_difference(b))
    return diff  # 如果没有差异则返回0.0


# In[4]:
gly_data = pd.read_csv('./all_gly.csv')
gly_data = pd.concat([gly_data, pd.read_csv('./vac/vac_gly.csv')])
gly_seqs = list(gly_data['seqs'])
ngly = list(gly_data['ngly'])

gly_result = {}
for i in range(len(gly_seqs)):
    gly_result[gly_seqs[i]] = eval(ngly[i])
data['N-Glycosylation'] = data.parallel_apply(calculate_gly_difference, axis=1,
                                              gly_result=gly_result)

properties_indexs = {
    'CHAM830107': {'A': 0.0, 'R': 0.0, 'N': 1.0, 'D': 1.0, 'C': 0.0, 'Q': 0.0, 'E': 1.0, 'G': 1.0, 'H': 0.0, 'I': 0.0,
                   'L': 0.0, 'K': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0},
    'RADA880108': {'A': -0.06, 'R': -0.84, 'N': -0.48, 'D': -0.8, 'C': 1.36, 'Q': -0.73, 'E': -0.77, 'G': -0.41,
                   'H': 0.49, 'I': 1.31, 'L': 1.21, 'K': -1.18, 'M': 1.27, 'F': 1.27, 'P': 0.0, 'S': -0.5, 'T': -0.27,
                   'W': 0.88, 'Y': 0.33, 'V': 1.09},
    'CIDH920101': {'A': -0.45, 'R': -0.24, 'N': -0.2, 'D': -1.52, 'C': 0.79, 'Q': -0.99, 'E': -0.8, 'G': -1.0,
                   'H': 1.07, 'I': 0.76, 'L': 1.29, 'K': -0.36, 'M': 1.37, 'F': 1.48, 'P': -0.12, 'S': -0.98, 'T': -0.7,
                   'W': 1.38, 'Y': 1.49, 'V': 1.26},
    'CHOC760102': {'A': 25.0, 'R': 90.0, 'N': 63.0, 'D': 50.0, 'C': 19.0, 'Q': 71.0, 'E': 49.0, 'G': 23.0, 'H': 43.0,
                   'I': 18.0, 'L': 23.0, 'K': 97.0, 'M': 31.0, 'F': 24.0, 'P': 50.0, 'S': 44.0, 'T': 47.0, 'W': 32.0,
                   'Y': 60.0, 'V': 18.0},
    'COHE430101': {'A': 0.75, 'R': 0.7, 'N': 0.61, 'D': 0.6, 'C': 0.61, 'Q': 0.67, 'E': 0.66, 'G': 0.64, 'H': 0.67,
                   'I': 0.9, 'L': 0.9, 'K': 0.82, 'M': 0.75, 'F': 0.77, 'P': 0.76, 'S': 0.68, 'T': 0.7, 'W': 0.74,
                   'Y': 0.71, 'V': 0.86}}
for i in properties_indexs:
    properties_indexs[i]['X'] = sum(list(properties_indexs[i].values())) / 20
    properties_indexs[i]['B'] = (properties_indexs[i]['D'] + properties_indexs[i]['N']) / 2

for item in tqdm(properties_indexs):
    data[item] = data.parallel_apply(calculate_property_difference, axis=1,
                                     property_dict=properties_indexs[item])

space_data = pd.read_csv('../0_preparation/mean_distances.csv', index_col='Residue_ID')
space_dict = space_data.to_dict()['Mean_Distance']
data['Distance'] = data.parallel_apply(calculate_space_difference, axis=1,
                                       space_dict=space_dict, ref_seq=seqs['>a/hong_kong/1/68|epi_isl_151|1968-01-01'])

epitopes_index = {
    'N2_A': [82, 83, 84, 86, 88, 89, 90, 187, 207, 208, 234, 236, 258, 259, 283, 284, 285, 286, 306, 307, 308, 309, 311,
             357, 415, 416],
    'N2_B': [118, 143, 146, 147, 150, 151, 152, 153, 154, 156, 178, 406, 430, 431, 432, 433, 434, 437, 469],
    'N2_C': [195, 196, 197, 198, 199, 200, 218, 219, 220, 221, 222, 224, 244, 245, 246, 247, 248, 249, 250, 251, 253,
             268, 273, 274, 276, 277],
    'N2_D': [292, 294, 295, 296, 326, 327, 328, 329, 331, 332, 333, 334, 339, 342, 343, 344, 345, 346, 347, 348, 369,
             370, 371],
    'N2_E': [392, 394, 399, 400, 401, 402, 453, 454, 455, 456, 457, 458]
}

ref_seq = seqs['>a/hong_kong/1/68|epi_isl_151|1968-01-01']
epitopes_index_move = {}
for i in epitopes_index:
    epitopes_index_move[i] = []
    for j in epitopes_index[i]:
        epitopes_index_move[i].append(j + ref_seq[:j].count('-'))

for epitope in tqdm(epitopes_index):
    data[epitope] = data.parallel_apply(calculate_epitope_difference, axis=1,
                                        epitope_sites=epitopes_index_move[epitope])

columns = list(data)
columns.remove('seq_a')
columns.remove('seq_b')
data[columns].to_csv('.//features.csv', index=None)

rf = joblib.load('..//model_construction/model/Random Forest.pkl')

feature_name = list(rf.feature_names_in_)
for i in feature_name:
    if i not in list(data):
        print(i)

x = data[feature_name]
scaler = joblib.load('../model_construction/model/scale.pkl')
x = pd.DataFrame(scaler.transform(data[feature_name]), columns=feature_name)
y = rf.predict_proba(x)

data['result'] = np.log(y[:, 0] / y[:, 1])
data[['strain_a', 'strain_b', 'result']].to_csv('./predict_result.csv', index=None)
new_data = data[data['result'] > 0]
new_data[['strain_a', 'strain_b', 'result']].to_csv('.//rf.abc', index=None, sep='\t')
