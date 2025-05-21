import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


def get_alpha_carbons(structure, chain_id='A'):
    '''
    Get Coordinates of CA
    '''
    alpha_carbons = {}
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if 'CA' in residue:
                        alpha_carbons[residue.get_id()[1]] = residue['CA'].get_coord()
    return alpha_carbons


def calculate_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)


def calculate_mean_distances(alpha_carbons, enzyme_sites):
    '''
    calculate_mean_distances to Catalyze sites
    '''
    distances = {}
    site_coords = [alpha_carbons[site] for site in enzyme_sites if site in alpha_carbons]
    for res_id, ca_coord in alpha_carbons.items():
        # distances to each  sites
        if len(site_coords) > 0:
            dist_list = [calculate_distance(ca_coord, site_coord) for site_coord in site_coords]
            mean_distance = np.mean(dist_list)
            distances[res_id] = mean_distance
    return distances


def save_to_csv(distances, output_file):
    df = pd.DataFrame(list(distances.items()), columns=['Residue_ID', 'Mean_Distance'])
    df.to_csv(output_file, index=False)


enzyme_sites = [118, 151, 152, 224, 276, 292, 371, 406, ]
# R118、D151、R152、R224、E276、R292、R371  Y406

# get pdb
pdb_file = '../epitope/7u4e.pdb'
output_file = 'mean_distances.csv'
parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', pdb_file)

alpha_carbons = get_alpha_carbons(structure, chain_id='A')
distances = calculate_mean_distances(alpha_carbons, enzyme_sites)
save_to_csv(distances, output_file)
print(f"结果已保存到 {output_file}")
