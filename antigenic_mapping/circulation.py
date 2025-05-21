import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42

# get cluster info for sampling strains
seqs_to_group = {}
group_num = 1
with open('./mcl_result/out.I15') as file:
    for line in file:
        strains = line.strip('\n').split('\t')
        if len(strains) >= 30:
            for i in strains:
                seqs_to_group[i] = group_num
            group_num += 1

seqs = {}
with open('./na_seq/na_seqs_mafft_normal.fasta', encoding='utf-8', errors='replace') as file:
    seqs = {}
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

sequence_to_group = {}
for i in seqs:
    if i in seqs_to_group:
        sequence_to_group[seqs[i]] = seqs_to_group[i]

seqs = {}
with open('./na_seq/na_all_seqs.fasta', encoding='utf-8', errors='replace') as file:
    seqs = {}
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

all_name_to_group = seqs_to_group.copy()
for i in seqs:
    if seqs[i] in sequence_to_group:
        all_name_to_group[i] = sequence_to_group[seqs[i]]

name = [i.split('|')[0][1:].replace('>', '') for i in all_name_to_group.keys()]
cluster = all_name_to_group.values()

c = {
    'strain': name,
    'cluster': cluster
}
data = pd.DataFrame(c)
data.to_csv('./all_seq_cluster.csv', index=False)

# get meta info
seq_data = pd.read_csv('./all_seq_cluster.csv')
seq = list(seq_data['strain'])

meta_data = pd.read_csv('./source_data/h3n2.csv')
meta_data = meta_data[meta_data['Isolate_Id'].isin(seq)]
meta_data['continent'] = [i.split('/')[0].strip() for i in meta_data['Location']]
meta_data['date'] = [i[:4] for i in meta_data['Collection_Date']]
meta_data = meta_data[meta_data['date'] > '1967']
data = pd.merge(seq_data, meta_data, left_on='strain', right_on='Isolate_Id')
data.to_csv('cluster_circulation.csv', index=False)
