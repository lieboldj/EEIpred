#%%
import numpy as np
import os
import csv
import pickle
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
#%%

def read_residue_positions(pos_file):
    if not os.path.exists(pos_file):
        return
    with open(pos_file, 'rt') as fh:
        pos = [ int(_) for _ in fh.readline().strip().split() ]
    pos = np.array(pos, dtype=np.int64)
    return pos

def get_labels():
    pairs = {}
    filename = f"../data/int_exon_pairs.txt"
    with open(filename, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            exon_name = row['exon1'] + "_" + row['exon2']
            exon_name = (row['exon1'],row['exon2'])
            if exon_name not in pairs:
                pairs[exon_name] = 1
    return pairs
#%%
fold = 1
with open(f"../data/test/test_{fold}.txt", 'r') as fh:
    test_data = fh.read().splitlines()
# load test info
test_info = []
with open(f"../data/test/test_info_{fold}.txt", 'r') as fh:
    lines = [line.rstrip() for line in fh]
    for line in lines:
        test_info.append(line.split("\t"))
with open("../../data_collection/chain_info.txt") as f_tr:
    chain_info = f_tr.read().splitlines()
    chain_info = [line.split("\t") for line in chain_info]
exon_dir = "../../data_collection/uniprot_EnsemblExonPDB_map"

inter_pairs = get_labels()

#%%
all_pos = 0
all_neg = 0
all_pos_exon = []
all_neg_exon = []

for i, sample in enumerate(test_data):
    protein1 = test_info[i][0]
    protein2 = test_info[i][1]
    pdb1 = sample[:-2]
    pdb2 = sample[:-4]+sample[-2:]

    try:
        if os.path.exists(f"../ckpts/test{fold}/score_{pdb1}:{pdb2}.npy"):
            scores = np.load(f"../ckpts/test{fold}/score_{pdb1}:{pdb2}.npy")
        else:
            scores = np.load(f"../ckpts/test{fold}/score_{pdb2}:{pdb1}.npy")
    except:
        print(f"Error: Could not load {pdb1}:{pdb2}")
        continue
    if os.path.exists(f"../examples/PDB/{pdb1}:{pdb2}/"\
                      +f"{pdb1}:{pdb2}.pkl"):
        if os.path.exists(f"../examples/PDB/{pdb1}:{pdb2}/"\
                      +f"labels_residues.pkl"):
            with open(f"../examples/PDB/{pdb1}:{pdb2}/"\
                      +f"labels_residues.pkl", "rb") as f:
                labels_res = pickle.load(f)
                sum_res = np.sum(labels_res)
            with open(f"../examples/PDB/{pdb1}:{pdb2}/"\
                      +f"labels.pkl", "rb") as f:
                dist = pickle.load(f)
                labels = (dist < 6).type(torch.float32)
                sum_lab = torch.sum(labels)
            if labels.shape != labels_res.shape:
                #print("labels does not fit: ", pdb1, pdb2)
                continue
            if sum_lab != sum_res:
                #print("labels does not fit sum: ", pdb1, pdb2)
                continue

    pos1 = read_residue_positions(f'../examples/PDB/'\
            f'{pdb1}:{pdb2}/{pdb1}/{pdb1}.pos')
    pos2 = read_residue_positions(f'../examples/PDB/' \
            f'{pdb1}:{pdb2}/{pdb2}/{pdb2}.pos')
    # check if number of residues in prediction matches number in pdb
    if scores.shape[0] != len(pos1) or scores.shape[1] != len(pos2):
        print("Error: Number of residues in prediction does not match number in pdb 1")
        print(f"{pdb1}:{pdb2}")
        continue

    if os.path.exists(f"{exon_dir}/{protein1}_{pdb1}.txt"):
        data1 = np.genfromtxt(f"{exon_dir}/{protein1}_{pdb1}.txt", delimiter="\t", dtype=str, skip_header=1)
    else:
        pdb_id_chain = [line[0] + "_" + line[2] for line in chain_info]
        idx = pdb_id_chain.index(pdb1)
        pdb1 = chain_info[idx][0] + "_" + chain_info[idx][1]
        data1 = np.genfromtxt(f"{exon_dir}/{protein1}_{pdb1}.txt", delimiter="\t", dtype=str, skip_header=1)
    # create dict which residue belongs to which exon from 1
    exon_dict1 = {int(line[-1]): line[0] for line in data1 if line[-1] != "-"}
    exons1 = {v: [k for k, val in exon_dict1.items() if val == v] for v in set(exon_dict1.values())}
    # load exon2 information
    if os.path.exists(f"{exon_dir}/{protein2}_{pdb2}.txt"):
        data2 = np.genfromtxt(f"{exon_dir}/{protein2}_{pdb2}.txt", delimiter="\t", dtype=str, skip_header=1)
    else:
        pdb_id_chain = [line[0] + "_" + line[2] for line in chain_info]
        idx = pdb_id_chain.index(pdb2)
        pdb2 = chain_info[idx][0] + "_" + chain_info[idx][1]
        data2 = np.genfromtxt(f"{exon_dir}/{protein2}_{pdb2}.txt", delimiter="\t", dtype=str, skip_header=1)
    # create dict which residue belongs to which exon from 2
    exon_dict2 = {int(line[-1]): line[0] for line in data2 if line[-1] != "-"}
    exons2 = {v: [k for k, val in exon_dict2.items() if val == v] for v in set(exon_dict2.values())}

    # from pos1/pos2 to index as dictionary as fast lookup
    pos1_dict = {value: index for index, value in enumerate(pos1)}
    pos2_dict = {value: index for index, value in enumerate(pos2)}

    # remove values from exons1 and exons2 which are in pdb but not in pos1/pos2
    for exon in list(exons1.keys()):
        exons1[exon] = [value for value in exons1[exon] if value in pos1_dict]
        if len(exons1[exon]) == 0:
            del exons1[exon]
    for exon in list(exons2.keys()):
        exons2[exon] = [value for value in exons2[exon] if value in pos2_dict]
        if len(exons2[exon]) == 0:
            del exons2[exon]
        

    # if we evaluate directly we do not need to write it into a dict
    exon_pairs = dict()
    for exon_1 in exons1:
        for exon_2 in exons2:
            index = [pos1_dict[value] for value in exons1[exon_1]]
            index2 = [pos2_dict[value] for value in exons2[exon_2]]

            exon_pairs[(exon_1, exon_2)] = scores[index, :][:,index2]
            #print(np.max(exon_pairs[(exon_1, exon_2)]))
            # if we want to evaluate with further metrices
            sorted_values = np.sort(exon_pairs[(exon_1, exon_2)].flatten())[::-1]   
            top_indices = np.arange(len(sorted_values)) < len(sorted_values) * 0.05

            if (exon_1, exon_2) in inter_pairs:
                all_pos += 1
                #all_pos_exon.append(np.mean(sorted_values[top_indices]))
                all_pos_exon.append(np.max(exon_pairs[(exon_1, exon_2)]))
            else:
                all_neg += 1
                #all_neg_exon.append(np.mean(sorted_values[top_indices]))
                all_neg_exon.append(np.max(exon_pairs[(exon_1, exon_2)]))
#print(all_pos, all_neg)

# %%
labels = np.concatenate((np.ones(all_pos), np.zeros(all_neg)))
roc_auc_score(labels, np.concatenate((all_pos_exon, all_neg_exon)))
# %%

#plt.hist(all_pos_exon, bins=100, alpha=0.5, label="positive")
#plt.hist(all_neg_exon, bins=100, alpha=0.5, label="negative")
#plt.xlabel("Maximum Glinter score of exon pairs in fold 1")
#plt.ylabel("Count")
#plt.legend()
# %%
