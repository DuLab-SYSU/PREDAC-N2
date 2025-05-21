import pandas as pd

data = pd.read_csv('../rf.abc', sep='\t')
groups = {}
a = 1
with open('./mcl_result/out.I15') as file:
    for line in file:
        strains = line.strip('\n').split('\t')
        if len(strains) > 30:
            print(a, '------')
            for i in strains:
                groups[i] = a
            a += 1

seqs = {}
with open('./na_seq/na_seqs_mafft_normal_maffted.fasta', encoding='utf-8', errors='replace') as file:
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

with open('./tree/seq_rename_rf.fasta', 'w') as file:
    for seq in groups:
        file.write(seq + f'_group_{groups[seq]}\n' + seqs[seq] + '\n')

import os

os.system('mafft --thread 10 --auto --inputorder ./seq_rename_rf.fasta > seq_rename_rf_mafft.fasta')
os.system('fasttree  -gamma -boot 1000  seq_rename_rf_mafft.fasta > rf.tree')
