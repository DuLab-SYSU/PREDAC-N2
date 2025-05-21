from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

data = pd.read_csv('../data_preparation/NAI/data.csv')
seq_a = list(data['seq_a'])
seq_b = list(data['seq_b'])
y = list(data['similarity'])
ref_seq = seq_a[0]

# caculate feature: epitopes
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


def calculate_epitope_difference(seq1, seq2, epitope_sites, ref_seq):
    epitope_seq1 = [seq1[i - 1 + ref_seq[:i].count('-')] for i in epitope_sites]
    epitope_seq2 = [seq2[i - 1 + ref_seq[:i].count('-')] for i in epitope_sites]
    return sum([a != b for a, b in zip(epitope_seq1, epitope_seq2)])


features_names = list(epitopes_index.keys())
epitopes = pd.DataFrame(columns=features_names)
for seq in range(len(seq_a)):
    seq1 = seq_a[seq]
    seq2 = seq_b[seq]
    feature_per_pairs = {}
    for key in features_names:
        feature_per_pairs[key] = calculate_epitope_difference(seq1, seq2, epitopes_index[key], ref_seq)
    epitopes = epitopes._append(feature_per_pairs, ignore_index=True)
epitopes.corr()


# caculate feature: property
def calculate_property_difference(seq1, seq2, property_dict):
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
            elif (aa1 == 'X') and (aa2 in property_dict):
                property_difference = abs(sum(property_dict.values()) / 20 - property_dict[aa2])
                differences.append(property_difference)
            elif (aa2 == 'X') and (aa1 in property_dict):
                property_difference = abs(sum(property_dict.values()) / 20 - property_dict[aa1])
                differences.append(property_difference)
            else:
                print(aa1, aa2)
    max_differences = sorted(differences, reverse=True)[:3]
    if max_differences:
        return sum(max_differences) / len(max_differences)
    else:
        return 0.0


## candidates
property_indexs = {
    'CHAM830107': {'A': 0.0, 'R': 0.0, 'N': 1.0, 'D': 1.0, 'C': 0.0, 'Q': 0.0, 'E': 1.0, 'G': 1.0, 'H': 0.0, 'I': 0.0,
                   'L': 0.0, 'K': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0},
    'CHAM830108': {'A': 0.0, 'R': 1.0, 'N': 1.0, 'D': 0.0, 'C': 1.0, 'Q': 1.0, 'E': 0.0, 'G': 0.0, 'H': 1.0, 'I': 0.0,
                   'L': 0.0, 'K': 1.0, 'M': 1.0, 'F': 1.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 1.0, 'Y': 1.0, 'V': 0.0},
    'FAUJ880111': {'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': 0.0, 'C': 0.0, 'Q': 0.0, 'E': 0.0, 'G': 0.0, 'H': 1.0, 'I': 0.0,
                   'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0},
    'KLEP840101': {'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0, 'Q': 0.0, 'E': -1.0, 'G': 0.0, 'H': 0.0, 'I': 0.0,
                   'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0},
    'GRAR740102': {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4,
                   'I': 5.2, 'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2,
                   'V': 5.9},
    'RADA880108': {'A': -0.06, 'R': -0.84, 'N': -0.48, 'D': -0.8, 'C': 1.36, 'Q': -0.73, 'E': -0.77, 'G': -0.41,
                   'H': 0.49, 'I': 1.31, 'L': 1.21, 'K': -1.18, 'M': 1.27, 'F': 1.27, 'P': 0.0, 'S': -0.5, 'T': -0.27,
                   'W': 0.88, 'Y': 0.33, 'V': 1.09},
    'ZIMJ680103': {'A': 0.0, 'R': 52.0, 'N': 3.38, 'D': 49.7, 'C': 1.48, 'Q': 3.53, 'E': 49.9, 'G': 0.0, 'H': 51.6,
                   'I': 0.13, 'L': 0.13, 'K': 49.5, 'M': 1.43, 'F': 0.35, 'P': 1.58, 'S': 1.67, 'T': 1.66, 'W': 2.1,
                   'Y': 1.61, 'V': 0.13},
    'ARGP820101': {'A': 0.61, 'R': 0.6, 'N': 0.06, 'D': 0.46, 'C': 1.07, 'Q': 0.0, 'E': 0.47, 'G': 0.07, 'H': 0.61,
                   'I': 2.22, 'L': 1.53, 'K': 1.15, 'M': 1.18, 'F': 2.02, 'P': 1.95, 'S': 0.05, 'T': 0.05, 'W': 2.65,
                   'Y': 1.88, 'V': 1.32},
    'CIDH920101': {'A': -0.45, 'R': -0.24, 'N': -0.2, 'D': -1.52, 'C': 0.79, 'Q': -0.99, 'E': -0.8, 'G': -1.0,
                   'H': 1.07, 'I': 0.76, 'L': 1.29, 'K': -0.36, 'M': 1.37, 'F': 1.48, 'P': -0.12, 'S': -0.98, 'T': -0.7,
                   'W': 1.38, 'Y': 1.49, 'V': 1.26},
    'EISD840101': {'A': 0.25, 'R': -1.76, 'N': -0.64, 'D': -0.72, 'C': 0.04, 'Q': -0.69, 'E': -0.62, 'G': 0.16,
                   'H': -0.4, 'I': 0.73, 'L': 0.53, 'K': -1.1, 'M': 0.26, 'F': 0.61, 'P': -0.07, 'S': -0.26, 'T': -0.18,
                   'W': 0.37, 'Y': 0.02, 'V': 0.54},
    'WILM950101': {'A': 0.06, 'R': -0.85, 'N': 0.25, 'D': -0.2, 'C': 0.49, 'Q': 0.31, 'E': -0.1, 'G': 0.21, 'H': -2.24,
                   'I': 3.48, 'L': 3.5, 'K': -1.62, 'M': 0.21, 'F': 4.8, 'P': 0.71, 'S': -0.62, 'T': 0.65, 'W': 2.29,
                   'Y': 1.89, 'V': 1.59},
    'BLAS910101': {'A': 0.616, 'R': 0.0, 'N': 0.236, 'D': 0.028, 'C': 0.68, 'Q': 0.251, 'E': 0.043, 'G': 0.501,
                   'H': 0.165, 'I': 0.943, 'L': 0.943, 'K': 0.283, 'M': 0.738, 'F': 1.0, 'P': 0.711, 'S': 0.359,
                   'T': 0.45, 'W': 0.878, 'Y': 0.88, 'V': 0.825},
    'JURD980101': {'A': 1.1, 'R': -5.1, 'N': -3.5, 'D': -3.6, 'C': 2.5, 'Q': -3.68, 'E': -3.2, 'G': -0.64, 'H': -3.2,
                   'I': 4.5, 'L': 3.8, 'K': -4.11, 'M': 1.9, 'F': 2.8, 'P': -1.9, 'S': -0.5, 'T': -0.7, 'W': -0.46,
                   'Y': -1.3, 'V': 4.2},
    'CHOC760101': {'A': 115.0, 'R': 225.0, 'N': 160.0, 'D': 150.0, 'C': 135.0, 'Q': 180.0, 'E': 190.0, 'G': 75.0,
                   'H': 195.0, 'I': 175.0, 'L': 170.0, 'K': 200.0, 'M': 185.0, 'F': 210.0, 'P': 145.0, 'S': 115.0,
                   'T': 140.0, 'W': 255.0, 'Y': 230.0, 'V': 155.0},
    'CHOC760102': {'A': 25.0, 'R': 90.0, 'N': 63.0, 'D': 50.0, 'C': 19.0, 'Q': 71.0, 'E': 49.0, 'G': 23.0, 'H': 43.0,
                   'I': 18.0, 'L': 23.0, 'K': 97.0, 'M': 31.0, 'F': 24.0, 'P': 50.0, 'S': 44.0, 'T': 47.0, 'W': 32.0,
                   'Y': 60.0, 'V': 18.0},
    'JANJ780101': {'A': 27.8, 'R': 94.7, 'N': 60.1, 'D': 60.6, 'C': 15.5, 'Q': 68.7, 'E': 68.2, 'G': 24.5, 'H': 50.7,
                   'I': 22.8, 'L': 27.6, 'K': 103.0, 'M': 33.5, 'F': 25.5, 'P': 51.5, 'S': 42.0, 'T': 45.0, 'W': 34.7,
                   'Y': 55.2, 'V': 23.7},
    'RADA880106': {'A': 93.7, 'R': 250.4, 'N': 146.3, 'D': 142.6, 'C': 135.2, 'Q': 177.7, 'E': 182.9, 'G': 52.6,
                   'H': 188.1, 'I': 182.2, 'L': 173.7, 'K': 215.2, 'M': 197.6, 'F': 228.6, 'P': 0.0, 'S': 109.5,
                   'T': 142.1, 'W': 271.6, 'Y': 239.9, 'V': 157.2},
    'ROSG850101': {'A': 86.6, 'R': 162.2, 'N': 103.3, 'D': 97.8, 'C': 132.3, 'Q': 119.2, 'E': 113.9, 'G': 62.9,
                   'H': 155.8, 'I': 158.0, 'L': 164.1, 'K': 115.5, 'M': 172.9, 'F': 194.1, 'P': 92.9, 'S': 85.6,
                   'T': 106.5, 'W': 224.6, 'Y': 177.7, 'V': 141.0},
    'ROSG850102': {'A': 0.74, 'R': 0.64, 'N': 0.63, 'D': 0.62, 'C': 0.91, 'Q': 0.62, 'E': 0.62, 'G': 0.72, 'H': 0.78,
                   'I': 0.88, 'L': 0.85, 'K': 0.52, 'M': 0.85, 'F': 0.88, 'P': 0.64, 'S': 0.66, 'T': 0.7, 'W': 0.85,
                   'Y': 0.76, 'V': 0.86},
    'BIGC670101': {'A': 52.6, 'R': 109.1, 'N': 75.7, 'D': 68.4, 'C': 68.3, 'Q': 89.7, 'E': 84.7, 'G': 36.3, 'H': 91.9,
                   'I': 102.0, 'L': 102.0, 'K': 105.1, 'M': 97.7, 'F': 113.9, 'P': 73.6, 'S': 54.9, 'T': 71.2,
                   'W': 135.4, 'Y': 116.2, 'V': 85.1},
    'BULH740102': {'A': 0.691, 'R': 0.728, 'N': 0.596, 'D': 0.558, 'C': 0.624, 'Q': 0.649, 'E': 0.632, 'G': 0.592,
                   'H': 0.646, 'I': 0.809, 'L': 0.842, 'K': 0.767, 'M': 0.709, 'F': 0.756, 'P': 0.73, 'S': 0.594,
                   'T': 0.655, 'W': 0.743, 'Y': 0.743, 'V': 0.777},
    'CHOC750101': {'A': 91.5, 'R': 202.0, 'N': 135.2, 'D': 124.5, 'C': 117.7, 'Q': 161.1, 'E': 155.1, 'G': 66.4,
                   'H': 167.3, 'I': 168.8, 'L': 167.9, 'K': 171.3, 'M': 170.8, 'F': 203.4, 'P': 129.3, 'S': 99.1,
                   'T': 122.1, 'W': 237.6, 'Y': 203.6, 'V': 141.7},
    'COHE430101': {'A': 0.75, 'R': 0.7, 'N': 0.61, 'D': 0.6, 'C': 0.61, 'Q': 0.67, 'E': 0.66, 'G': 0.64, 'H': 0.67,
                   'I': 0.9, 'L': 0.9, 'K': 0.82, 'M': 0.75, 'F': 0.77, 'P': 0.76, 'S': 0.68, 'T': 0.7, 'W': 0.74,
                   'Y': 0.71, 'V': 0.86},
    'FAUJ880103': {'A': 1.0, 'R': 6.13, 'N': 2.95, 'D': 2.78, 'C': 2.43, 'Q': 3.95, 'E': 3.78, 'G': 0.0, 'H': 4.66,
                   'I': 4.0, 'L': 4.0, 'K': 4.77, 'M': 4.43, 'F': 5.89, 'P': 2.72, 'S': 1.6, 'T': 2.6, 'W': 8.08,
                   'Y': 6.47, 'V': 3.0},
    'GOLD730102': {'A': 88.3, 'R': 181.2, 'N': 125.1, 'D': 110.8, 'C': 112.4, 'Q': 148.7, 'E': 140.5, 'G': 60.0,
                   'H': 152.6, 'I': 168.5, 'L': 168.5, 'K': 175.6, 'M': 162.2, 'F': 189.0, 'P': 122.2, 'S': 88.7,
                   'T': 118.2, 'W': 227.0, 'Y': 193.0, 'V': 141.4},
    'GRAR740103': {'A': 31.0, 'R': 124.0, 'N': 56.0, 'D': 54.0, 'C': 55.0, 'Q': 85.0, 'E': 83.0, 'G': 3.0, 'H': 96.0,
                   'I': 111.0, 'L': 111.0, 'K': 119.0, 'M': 105.0, 'F': 132.0, 'P': 32.5, 'S': 32.0, 'T': 61.0,
                   'W': 170.0, 'Y': 136.0, 'V': 84.0},
    'KRIW790103': {'A': 27.5, 'R': 105.0, 'N': 58.7, 'D': 40.0, 'C': 44.6, 'Q': 80.7, 'E': 62.0, 'G': 0.0, 'H': 79.0,
                   'I': 93.5, 'L': 93.5, 'K': 100.0, 'M': 94.1, 'F': 115.5, 'P': 41.9, 'S': 29.3, 'T': 51.3, 'W': 145.5,
                   'Y': 117.3, 'V': 71.5},
    'TSAJ990101': {'A': 89.3, 'R': 190.3, 'N': 122.4, 'D': 114.4, 'C': 102.5, 'Q': 146.9, 'E': 138.8, 'G': 63.8,
                   'H': 157.5, 'I': 163.0, 'L': 163.1, 'K': 165.1, 'M': 165.8, 'F': 190.8, 'P': 121.6, 'S': 94.2,
                   'T': 119.6, 'W': 226.4, 'Y': 194.6, 'V': 138.2},
    'TSAJ990102': {'A': 90.0, 'R': 194.0, 'N': 124.7, 'D': 117.3, 'C': 103.3, 'Q': 149.4, 'E': 142.2, 'G': 64.9,
                   'H': 160.0, 'I': 163.9, 'L': 164.0, 'K': 167.3, 'M': 167.0, 'F': 191.9, 'P': 122.9, 'S': 95.4,
                   'T': 121.5, 'W': 228.2, 'Y': 197.0, 'V': 139.0},
    'HARY940101': {'A': 90.1, 'R': 192.8, 'N': 127.5, 'D': 117.1, 'C': 113.2, 'Q': 149.4, 'E': 140.8, 'G': 63.8,
                   'H': 159.3, 'I': 164.9, 'L': 164.6, 'K': 170.0, 'M': 167.7, 'F': 193.5, 'P': 123.1, 'S': 94.2,
                   'T': 120.0, 'W': 197.1, 'Y': 231.7, 'V': 139.1},
    'FAUJ880109': {'A': 0.0, 'R': 4.0, 'N': 2.0, 'D': 1.0, 'C': 0.0, 'Q': 2.0, 'E': 1.0, 'G': 0.0, 'H': 1.0, 'I': 0.0,
                   'L': 0.0, 'K': 2.0, 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 1.0, 'T': 1.0, 'W': 1.0, 'Y': 1.0, 'V': 0.0},
    'BURA740101': {'A': 0.486, 'R': 0.262, 'N': 0.193, 'D': 0.288, 'C': 0.2, 'Q': 0.418, 'E': 0.538, 'G': 0.12,
                   'H': 0.4, 'I': 0.37, 'L': 0.42, 'K': 0.402, 'M': 0.417, 'F': 0.318, 'P': 0.208, 'S': 0.2, 'T': 0.272,
                   'W': 0.462, 'Y': 0.161, 'V': 0.379},
    'CHOP780202': {'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19, 'Q': 1.1, 'E': 0.37, 'G': 0.75, 'H': 0.87,
                   'I': 1.6, 'L': 1.3, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55, 'S': 0.75, 'T': 1.19, 'W': 1.37,
                   'Y': 1.47, 'V': 1.7},
    'CHOP780201': {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.7, 'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.0,
                   'I': 1.08, 'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08,
                   'Y': 0.69, 'V': 1.06},
    'CRAJ730102': {'A': 1.0, 'R': 0.74, 'N': 0.75, 'D': 0.89, 'C': 0.99, 'Q': 0.87, 'E': 0.37, 'G': 0.56, 'H': 0.36,
                   'I': 1.75, 'L': 1.53, 'K': 1.18, 'M': 1.4, 'F': 1.26, 'P': 0.36, 'S': 0.65, 'T': 1.15, 'W': 0.84,
                   'Y': 1.41, 'V': 1.61},
    'ISOY800101': {'A': 1.53, 'R': 1.17, 'N': 0.6, 'D': 1.0, 'C': 0.89, 'Q': 1.27, 'E': 1.63, 'G': 0.44, 'H': 1.03,
                   'I': 1.07, 'L': 1.32, 'K': 1.26, 'M': 1.66, 'F': 1.22, 'P': 0.25, 'S': 0.65, 'T': 0.86, 'W': 1.05,
                   'Y': 0.7, 'V': 0.93},
    'KANM800102': {'A': 0.81, 'R': 0.85, 'N': 0.62, 'D': 0.71, 'C': 1.17, 'Q': 0.98, 'E': 0.53, 'G': 0.88, 'H': 0.92,
                   'I': 1.48, 'L': 1.24, 'K': 0.77, 'M': 1.05, 'F': 1.2, 'P': 0.61, 'S': 0.92, 'T': 1.18, 'W': 1.18,
                   'Y': 1.23, 'V': 1.66},
    'LEVM780101': {'A': 1.29, 'R': 0.96, 'N': 0.9, 'D': 1.04, 'C': 1.11, 'Q': 1.27, 'E': 1.44, 'G': 0.56, 'H': 1.22,
                   'I': 0.97, 'L': 1.3, 'K': 1.23, 'M': 1.47, 'F': 1.07, 'P': 0.52, 'S': 0.82, 'T': 0.82, 'W': 0.99,
                   'Y': 0.72, 'V': 0.91},
    'LEVM780102': {'A': 0.9, 'R': 0.99, 'N': 0.76, 'D': 0.72, 'C': 0.74, 'Q': 0.8, 'E': 0.75, 'G': 0.92, 'H': 1.08,
                   'I': 1.45, 'L': 1.02, 'K': 0.77, 'M': 0.97, 'F': 1.32, 'P': 0.64, 'S': 0.95, 'T': 1.21, 'W': 1.14,
                   'Y': 1.25, 'V': 1.49},
    'LEVM780104': {'A': 1.32, 'R': 0.98, 'N': 0.95, 'D': 1.03, 'C': 0.92, 'Q': 1.1, 'E': 1.44, 'G': 0.61, 'H': 1.31,
                   'I': 0.93, 'L': 1.31, 'K': 1.25, 'M': 1.39, 'F': 1.02, 'P': 0.58, 'S': 0.76, 'T': 0.79, 'W': 0.97,
                   'Y': 0.73, 'V': 0.93},
    'LEVM780105': {'A': 0.86, 'R': 0.97, 'N': 0.73, 'D': 0.69, 'C': 1.04, 'Q': 1.0, 'E': 0.66, 'G': 0.89, 'H': 0.85,
                   'I': 1.47, 'L': 1.04, 'K': 0.77, 'M': 0.93, 'F': 1.21, 'P': 0.68, 'S': 1.02, 'T': 1.27, 'W': 1.26,
                   'Y': 1.31, 'V': 1.43},
    'MAXF760101': {'A': 1.43, 'R': 1.18, 'N': 0.64, 'D': 0.92, 'C': 0.94, 'Q': 1.22, 'E': 1.67, 'G': 0.46, 'H': 0.98,
                   'I': 1.04, 'L': 1.36, 'K': 1.27, 'M': 1.53, 'F': 1.19, 'P': 0.49, 'S': 0.7, 'T': 0.78, 'W': 1.01,
                   'Y': 0.69, 'V': 0.98},
    'PALJ810103': {'A': 0.81, 'R': 1.03, 'N': 0.81, 'D': 0.71, 'C': 1.12, 'Q': 1.03, 'E': 0.59, 'G': 0.94, 'H': 0.85,
                   'I': 1.47, 'L': 1.03, 'K': 0.77, 'M': 0.96, 'F': 1.13, 'P': 0.75, 'S': 1.02, 'T': 1.19, 'W': 1.24,
                   'Y': 1.35, 'V': 1.44},
    'NAGK730101': {'A': 1.29, 'R': 0.83, 'N': 0.77, 'D': 1.0, 'C': 0.94, 'Q': 1.1, 'E': 1.54, 'G': 0.72, 'H': 1.29,
                   'I': 0.94, 'L': 1.23, 'K': 1.23, 'M': 1.23, 'F': 1.23, 'P': 0.7, 'S': 0.78, 'T': 0.87, 'W': 1.06,
                   'Y': 0.63, 'V': 0.97},
    'PALJ810104': {'A': 0.9, 'R': 0.75, 'N': 0.82, 'D': 0.75, 'C': 1.12, 'Q': 0.95, 'E': 0.44, 'G': 0.83, 'H': 0.86,
                   'I': 1.59, 'L': 1.24, 'K': 0.75, 'M': 0.94, 'F': 1.41, 'P': 0.46, 'S': 0.7, 'T': 1.2, 'W': 1.28,
                   'Y': 1.45, 'V': 1.73},
    'PALJ810101': {'A': 1.3, 'R': 0.93, 'N': 0.9, 'D': 1.02, 'C': 0.92, 'Q': 1.04, 'E': 1.43, 'G': 0.63, 'H': 1.33,
                   'I': 0.87, 'L': 1.3, 'K': 1.23, 'M': 1.32, 'F': 1.09, 'P': 0.63, 'S': 0.78, 'T': 0.8, 'W': 1.03,
                   'Y': 0.71, 'V': 0.95},
    'PRAM900103': {'A': 0.9, 'R': 0.99, 'N': 0.76, 'D': 0.72, 'C': 0.74, 'Q': 0.8, 'E': 0.75, 'G': 0.92, 'H': 1.08,
                   'I': 1.45, 'L': 1.02, 'K': 0.77, 'M': 0.97, 'F': 1.32, 'P': 0.64, 'S': 0.95, 'T': 1.21, 'W': 1.14,
                   'Y': 1.25, 'V': 1.49},
}

## calculate for all candidates
properties = pd.DataFrame(columns=property_indexs.keys())
for seq in tqdm(range(len(seq_a))):
    seq1 = seq_a[seq]
    seq2 = seq_b[seq]
    feature_per_pairs = {}

    for item in property_indexs:
        feature_per_pairs[item] = calculate_property_difference(seq1, seq2, property_indexs[item])
    properties = properties._append(feature_per_pairs, ignore_index=True)

## candidate group
charge_items = ['CHAM830107', 'CHAM830108', 'FAUJ880111', 'KLEP840101', ]
polar_items = ['GRAR740102', 'RADA880108', 'ZIMJ680103', ]
hydropho_items = ['ARGP820101', 'CIDH920101', 'EISD840101', 'WILM950101', 'BLAS910101', 'JURD980101', ]
access_items = ['CHOC760101', 'CHOC760102', 'JANJ780101', 'RADA880106', 'ROSG850101', 'ROSG850102', ]
volume_items = ['BIGC670101', 'BULH740102', 'CHOC750101', 'COHE430101', 'FAUJ880103', 'GOLD730102', 'GRAR740103',
                'KRIW790103', 'TSAJ990101', 'TSAJ990102', 'HARY940101', ]

## select by feature contribution
X_train, X_test, y_train, y_test = train_test_split(properties, y, test_size=0.3, random_state=0)
standard = StandardScaler()
X_train = pd.DataFrame(standard.fit_transform(X_train), columns=list(X_train))
max_contributing_features = []
random_forest = RandomForestClassifier(random_state=0)
all_items = [charge_items, polar_items, hydropho_items, access_items, volume_items]
for group_items in all_items:
    group_feature = X_train[group_items]
    random_forest.fit(group_feature, y_train)
    feature_importances = random_forest.feature_importances_
    feature_names = group_items
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_feature_importances = feature_importances[sorted_indices]
    print(f"Feature: {sorted_feature_names[0]}, Contribution: {sorted_feature_importances[0]}")
    max_contributing_features.append(sorted_feature_names[0])

## result
properties_indexs = {}
for i in max_contributing_features:
    properties_indexs.update({i: property_indexs[i]})
print(properties_indexs)
features_names = list(properties_indexs.keys())
properties = pd.DataFrame(columns=features_names)
for seq in range(len(seq_a)):
    seq1 = seq_a[seq]
    seq2 = seq_b[seq]
    feature_per_pairs = {}
    for key in features_names:
        feature_per_pairs[key] = calculate_property_difference(seq1, seq2, properties_indexs[key])
    properties = properties._append(feature_per_pairs, ignore_index=True)


# caculate feature: catalyze
def calculate_space_difference(seq1, seq2, space_dict, ref_seq):
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


space_data = pd.read_csv('../data_preparation/catalyze/mean_distances.csv', index_col='Residue_ID')
space_dict = space_data.to_dict()['Mean_Distance']
rbd_distance = []
for i in range(len(seq_a)):
    rbd_distance.append(calculate_space_difference(seq_a[i], seq_b[i], space_dict, ref_seq))


# caculate feature: glycosylation
def get_nglyco_sites(file_name):
    n_glyco_sites_num = defaultdict(list)
    with open(file_name) as file:
        content = []
        flag = 0
        for line in file:
            if flag % 3 == 2:
                content.append(line.strip())
            if line.startswith('------'):
                flag += 1
            if content and flag % 3 == 0:
                for site_content in content:
                    tem_data = site_content.split(' ')
                    while ' ' in tem_data:
                        tem_data.remove(' ')
                    while '' in tem_data:
                        tem_data.remove('')
                    if '+' in tem_data[-1]:
                        tem_name = '>' + tem_data[0]
                        n_glyco_sites_num[tem_name].append(int(tem_data[1]))
                content = []
    '''# avoid 0, but won't occur; otherwise error
    for seq in input_names.keys():
        if seq not in n_glyco_sites_num:
            n_glyco_sites_num[seq] = 0'''
    return dict(n_glyco_sites_num)


n_result = {}
n_result.update(get_nglyco_sites('./gly_experiment.txt'))
with open('../data_preparation/NAI/experiment_in_num.fasta') as file:
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
gly_dict = {}
for i in seqs:
    gly_dict[seqs[i]] = n_result[i]


def calculate_gly_difference(strain1, strain2, gly_result):
    a = set(gly_result[strain1])
    b = set(gly_result[strain2])
    diff = len(a.symmetric_difference(b))
    return diff


seq_a = list(data['seq_a'])
seq_b = list(data['seq_b'])
gly = []
for i in range(len(seq_b)):
    gly.append(calculate_gly_difference(seq_a[i].replace('-', ''), seq_b[i].replace('-', ''), gly_dict))

# combine
feature_frame = pd.concat([data, epitopes], axis=1)
feature_frame = pd.concat([feature_frame, properties], axis=1)
feature_frame['Distance'] = rbd_distance
feature_frame['N-Glycosylation'] = gly
feature_frame.to_csv('data_feature.csv', index=None)
