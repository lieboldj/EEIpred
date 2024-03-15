import numpy as np
import os
from Bio.PDB import PDBParser
from tqdm import tqdm
import argparse
import time
start = time.time()

parser = argparse.ArgumentParser(description='glinter')
parser.add_argument('--fold', type=int, default=1, help='fold number')
parser.add_argument('--origin_path', type=str, default='/cosybio/project/EEIP/EEIP', help='origin path')
parser.add_argument('--exon_mapping_dir', type=str, default= '/data_collection/uniprot_EnsemblExonPDB_map', help='exon mapping directory')
parser.add_argument('--database', type=str, default='exon/', help='Database name')
parser.add_argument('--dataset', type=str, default='CONTACT', help='Dataset name')
parser.add_argument('--mode', type=str, default='test', help='train or test, default test')

args = parser.parse_args()
#1HE1
origin_path = args.origin_path + "/glinter"
dataset = args.dataset

fold = args.fold
mode = args.mode

# get all residues from both protein pdb files individually
def get_unique_residues(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    unique_residues = list()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()[1]
                unique_residues.append(residue_id)
    return unique_residues

with open(f"{args.origin_path}/data_collection/cv_splits/{dataset}/{mode}{fold}_glinter.txt", "r") as f:
    pdb_list = f.read().splitlines()
with open(f"{args.origin_path}/data_collection/cv_splits/{dataset}/{mode}_info{fold}_glinter.txt", "r") as f:
    train_list = f.read().splitlines()
out_dir = origin_path + f"/results/{dataset}/fold{fold}/{mode}"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for i, pdbs in enumerate(tqdm(pdb_list)):
    proteins = train_list[i].split("\t")
    protein1 = proteins[0]
    protein2 = proteins[1]
    #split pdb
    pdb = pdbs.split("_")
    pdb1 = pdb[0] + "_" + pdb[1]
    pdb2 = pdb[0] + "_" + pdb[2]

    pdb_path1 = f"{origin_path}/examples/PDB/{pdb1}.pdb"
    pdb_path2 = f"{origin_path}/examples/PDB/{pdb2}.pdb"
    scores1 = f"{origin_path}/ckpts/{dataset}/{mode}{fold}/score_{pdb1}:{pdb2}.npy"

    prot1 = np.load(scores1)
    exon_dir = args.origin_path + args.exon_mapping_dir

    if os.path.exists(f"{exon_dir}/{protein1}_{pdb1}.txt"):
        data1 = np.genfromtxt(f"{exon_dir}/{protein1}_{pdb1}.txt", delimiter="\t", dtype=str, skip_header=1)

    # create dict which residue belongs to which exon from 1
    exon_dict1 = {int(line[-1]): line[0] for line in data1 if line[-1] != "-"}
    exons1 = {v: [k for k, val in exon_dict1.items() if val == v] for v in set(exon_dict1.values())}
    # load exon2 information
    if os.path.exists(f"{exon_dir}/{protein2}_{pdb2}.txt"):
        data2 = np.genfromtxt(f"{exon_dir}/{protein2}_{pdb2}.txt", delimiter="\t", dtype=str, skip_header=1)

    # create dict which residue belongs to which exon from 2
    exon_dict2 = {int(line[-1]): line[0] for line in data2 if line[-1] != "-"}
    exons2 = {v: [k for k, val in exon_dict2.items() if val == v] for v in set(exon_dict2.values())}
    
    residues1_pdb = get_unique_residues(pdb_path1)
    residues2_pdb = get_unique_residues(pdb_path2)

   # print(len(residues1_pdb), len(residues2_pdb), prot1.shape)
    pdb_dict1 = dict(zip(residues1_pdb, np.arange(prot1.shape[0])))
    pdb_dict2 = dict(zip(residues2_pdb, np.arange(prot1.shape[1])))

    ex_values1 = dict()
    ex_values2 = dict()

    miss_exons1 = []
    miss_exons2 = []
    for ex in exons1:
        # if res is not in pdb_dict1, then remove it from exons1
        exons1[ex] = [res for res in exons1[ex] if res in pdb_dict1]
        if len(exons1[ex]) == 0:
            miss_exons1.append(ex)         
        ex_values1[ex] = [pdb_dict1[res] for res in exons1[ex]]
    for ex in exons2:
        exons2[ex] = [res for res in exons2[ex] if res in pdb_dict2]
        if len(exons2[ex]) == 0:
            miss_exons2.append(ex)
        ex_values2[ex] = [pdb_dict2[res] for res in exons2[ex]]

    #print(sum(len(v) for v in ex_values1.values()), sum(len(v) for v in ex_values2.values()))
    result = {}
    for k1, v1 in ex_values1.items():
        for k2, v2 in ex_values2.items():
            exon_pair = prot1[np.ix_(v1, v2)]
            save_matrix = np.array(exon_pair, dtype=np.float32)
            if save_matrix.shape[0] > 100 or save_matrix.shape[1] > 100:
                #bigs += 1
                np.save(f"{out_dir}/{protein1}_{protein2}_{pdb1}_{pdb2}_{k1}_{k2}_big.npy", save_matrix, allow_pickle=False)
            else:
                np.save(f"{out_dir}/{protein1}_{protein2}_{pdb1}_{pdb2}_{k1}_{k2}.npy", save_matrix, allow_pickle=False)

print(f"Time: {time.time() - start}")