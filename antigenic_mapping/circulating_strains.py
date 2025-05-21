import pandas as pd

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
