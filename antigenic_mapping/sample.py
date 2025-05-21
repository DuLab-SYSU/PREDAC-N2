import os
import random
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# sampling
data = pd.read_csv('./source_data/h3n2.csv')
data_filtered = data.dropna(subset=['HA Segment_Id', 'NA Segment_Id'])
print(data.shape, data_filtered.shape)

continent = [i.split('/')[0].strip() for i in data_filtered['Location']]
date = [i[:7] for i in data_filtered['Collection_Date']]
strain_acc = list(data_filtered['Isolate_Id'])

seqs_in_group = defaultdict(list)
for i in tqdm(range(len(continent))):
    place = continent[i]
    time = date[i]
    seq_info = time + '|' + place
    seqs_in_group[seq_info].append(strain_acc[i])

# even sample
seqs_in_group_sampled = {}
sample_number = 7
for key, value in seqs_in_group.items():
    if len(value) > sample_number:
        random.seed(0)
        seqs_in_group_sampled[key] = random.sample(value, sample_number)
    else:
        seqs_in_group_sampled[key] = value

'''
propotion sample:
random.seed(0)
seqs_in_group_sampled = {}
sample_percentage = 0.05
for key, value in seqs_in_group.items():
    sample_number = int(len(value) * sample_percentage) + 1
    seqs_in_group_sampled[key] = random.sample(value, sample_number)
sample_strain_acc = []
for i in seqs_in_group_sampled:
    sample_strain_acc.extend(seqs_in_group_sampled[i])
'''

sample_strain_acc = []
for i in seqs_in_group_sampled:
    sample_strain_acc.extend(seqs_in_group_sampled[i])

seqs = {}
with open('./source_data/h3n2.fasta') as file:
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

seqs_rename = {}
for i in seqs:
    parts = i.split('|')
    tem_strain = parts[0][1:].strip()
    tem_segment = parts[-1].strip()
    seqs_rename[tem_strain + '|' + tem_segment] = seqs[i]

ha_seqs = {}
na_seqs = {}
for i in sample_strain_acc:
    if (i + '|NA') not in seqs_rename:
        print(i + '|NA')
        continue
    elif (i + '|HA') not in seqs_rename:
        print(i + '|HA')
        continue
    else:
        na_seqs[i + '|NA'] = seqs_rename[i + '|NA']
        ha_seqs[i + '|HA'] = seqs_rename[i + '|HA']

with open('./vac/vac.fasta') as file:
    tem_seq = tem_name = ''
    for line in file:
        line = line.strip()
        if '>' in line:
            na_seqs[tem_name] = tem_seq
            tem_name = line[1:]
            tem_seq = ''
        else:
            tem_seq += line
na_seqs[tem_name] = tem_seq
na_seqs.pop('')

with open('./na_seq/na_seqs.fasta', 'w') as file:
    for seq in na_seqs:
        file.write(f'>{seq}\n{na_seqs[seq]}\n')

# remove duplicate and quantification control
ha_duplicate = {}
na_duplicate = {}
for i in ha_seqs:
    ha_duplicate[ha_seqs[i]] = i
for i in na_seqs:
    na_duplicate[na_seqs[i]] = i

with open('na_seq/na_duplicate.fasta', 'w') as file:
    for seq in na_duplicate:
        if (seq.count('X') < 3) and (len(seq) > 400):
            file.write(f'>{na_duplicate[seq]}\n{seq}\n')

os.system('"/usr/bin/mafft" --thread 28 --auto --inputorder "na_seq/na_duplicate.fasta" > "na_seq/na_seqs_mafft.fasta"')
seqs = {}
with open('na_seq/na_seqs_mafft.fasta', encoding='utf-8', errors='replace') as file:
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

ref = seqs['>a/hong_kong/1/68|epi_isl_151|1968-01-01']
seqs_list = list(seqs.values())
abnormal_sites = []
for i in range(len(ref)):
    if ref[i] == '-':
        count = 0
        for j in range(len(seqs_list)):
            if seqs_list[j][i] == '-':
                count += 1
        if count > len(seqs_list) * 0.99:
            abnormal_sites.append(i)

seqs_list = list(seqs.values())
abnormal_seqs = []
for seq in seqs_list:
    for site in abnormal_sites:
        if seq[site] != '-':
            abnormal_seqs.append(seq)
            continue

with open('./na_seq/na_seqs_mafft_normal.fasta', 'w') as file:
    for i in seqs:
        if seqs[i] not in abnormal_seqs:
            file.write(i + '\n')
            file.write(seqs[i].replace('-', '') + '\n')

os.system('"/usr/bin/mafft" --thread 28 --auto --inputorder "./na_seq/na_seqs_mafft_normal.fasta" > "./na_seq/na_seqs_mafft_normal_maffted.fasta"')
