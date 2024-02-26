#%%
from __future__ import print_function
import argparse
import os
import sys
sys.path.append(".")
sys.path.append("../")
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pinet.model import PointNetDenseCls12, feature_transform_regularizer
from Bio.PDB import PDBParser
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from getcontactEpipred import getcontactbyabag,getsppider2,getalign
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#%%
#1HE1
random.seed(random.randint(1, 10000) )
torch.manual_seed(random.randint(1, 10000) )
fold = int(sys.argv[1])
#pdb = sys.argv[3]
# read text1.txt to pdb_list
with open("../../../data_collection/chain_info.txt") as f_tr:
    chain_info = f_tr.read().splitlines()
    chain_info = [line.split("\t") for line in chain_info]

with open(f"../../../data_collection/lists/test{fold+1}.txt", "r") as f:
    pdb_list = f.read().splitlines()
with open(f"../../../data_collection/lists/test_info{fold+1}.txt", "r") as f:
    pdb_info = f.read().splitlines()
proteins1 = list()
proteins2 = list()
for infos in pdb_info:
    info = infos.split("\t")
    proteins1.append(info[0])
    proteins2.append(info[1])
not_working = list()
#pdb_list = ["1a02_F_J","6msb_C_V"]
true = list()
preds = list()
counter = 0
index = 0
for i, pdbs in enumerate(tqdm(pdb_list[index:])):
    i = i+index
    protein1 = proteins1[i]
    protein2 = proteins2[i]
    #pdbs = "1a02_F_J"
    #pdbs = "6msb_C_V"
    #split pdb
    pdb = pdbs.split("_")
    pdb1 = pdb[0] + "_" + pdb[1]
    pdb2 = pdb[0] + "_" + pdb[2]
    #filel=sys.argv[1]
    filel = f"../../data/exon/lf/points/{pdb1}.pts"
    #filer=sys.argv[2]
    filer = f"../../data/exon/rf/points/{pdb2}.pts"

    num_classes = 2
    classifier = PointNetDenseCls12(k=num_classes, feature_transform=False,pdrop=0.0,id=5)

    classifier.cuda()


    PATH=f'../seg/seg_model_protein_{fold}_19.pth'
    #PATH='../models/split_0mmgk.pth'
    classifier.load_state_dict(torch.load(PATH))
    classifier.eval()
    try:
        pointsr=np.loadtxt(filer).astype(np.float32)
        pointsl=np.loadtxt(filel).astype(np.float32)
    except:
        counter += 1
        not_working.append([protein1, protein2, pdb1, pdb2])
        continue
    #pointsr=np.loadtxt(filer).astype(np.float32)
    #pointsl=np.loadtxt(filel).astype(np.float32)

    coordsetr = pointsr[:, 0:3]
    featsetr = pointsr[:, 3:]

    coordsetl = pointsl[:, 0:3]
    featsetl = pointsl[:, 3:]

    featsetr = featsetr / np.sqrt(np.max(featsetr ** 2, axis=0))
    featsetl = featsetl / np.sqrt(np.max(featsetl ** 2, axis=0))

    coordsetr = coordsetr - np.expand_dims(np.mean(coordsetr, axis=0), 0)  # center
    coordsetl = coordsetl - np.expand_dims(np.mean(coordsetl, axis=0), 0)  # center

    pointsr[:, 0:5] = np.concatenate((coordsetr, featsetr), axis=1)
    pointsl[:, 0:5] = np.concatenate((coordsetl, featsetl), axis=1)

    pointsr=torch.from_numpy(pointsr).unsqueeze(0)
    pointsl=torch.from_numpy(pointsl).unsqueeze(0)


    memlim=120000
    if pointsl.size()[1] + pointsr.size()[1] > memlim:
        lr = pointsl.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
        rr = pointsr.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
        
        ls = np.random.choice(pointsl.size()[1], int(lr), replace=False)
        rs = np.random.choice(pointsr.size()[1], int(rr), replace=False)

        pointsr = pointsr[:, rs, :]
        pointsl = pointsl[:, ls, :]

    pointsr = pointsr.transpose(2, 1).cuda()
    pointsl = pointsl.transpose(2, 1).cuda()

    classifier = classifier.eval()

    pred, _, _ = classifier(pointsr,pointsl)

    pred = pred.view(-1, 1)

    rfile = filer#pdb + '-r.pts'
    rcoord = np.transpose(np.loadtxt(rfile))[0:3,:]

    lfile = filel#pdb + '-l.pts'
    lcoord = np.transpose(np.loadtxt(lfile))[0:3,:]


    prolabel = pred.to('cpu').detach().numpy()

    pror = prolabel[0:rcoord.shape[1]]
    prol = prolabel[rcoord.shape[1]:]


    rcoord = np.transpose(rcoord)
    lcoord = np.transpose(lcoord)

    nn = 3
    dt = 2
    #cutoff = 0.5
    #tol = [6, 6, 6]

    pdb_path1 = f"../../data/exon/pdb/{pdb1}.pdb"
    pdb_path2 = f"../../data/exon/pdb/{pdb2}.pdb"

    rl, nl, cl = getsppider2(pdb_path1)
    rr, nr, cr = getsppider2(pdb_path2)

    rr_1 = np.unique(rr)
    rl_1 = np.unique(rl)

    assert(len(rr_1) == len(cr))
    assert(len(rl_1) == len(cl))
    cencoordr = np.asarray(nr)

    cencoordl = np.asarray(nl)

    clfr = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(rcoord)
    distancesr, indicesr = clfr.kneighbors(cencoordr)

    probr = [0] * len(cr)
    for_loop = 0
    for ii,ind in enumerate(indicesr):
        for jj,sind in enumerate(ind):
            if distancesr[ii][jj]>dt:
                continue
            try: 
                probr[rr[ii]] = max(probr[rr[ii]], pror[sind])
            except:
                for_loop = 1
                continue
    if for_loop:
        not_working.append([protein1, protein2, pdb1, pdb2])
        counter += 1
        continue

    clfl = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(lcoord)
    distancesl, indicesl = clfl.kneighbors(cencoordl)

    probl = [0] * len(cl)
    for_loop = 0
    for ii,ind in enumerate(indicesl):
        for jj, sind in enumerate(ind):

            if distancesl[ii][jj]>dt:
                continue
            try: 
                probl[rl[ii]] = max(probl[rl[ii]],  prol[sind])
            except:
                for_loop = 1
                continue
    if for_loop:
        not_working.append([protein1, protein2, pdb1, pdb2])
        counter += 1
        continue

    prot1 = probl
    prot2 = probr
    #np.savetxt(pdb1+'_resi.seg',np.array(probl))
    #np.savetxt(pdb2+'_resi.seg',np.array(probr))


    exon_dir = "../../../data_collection/uniprot_EnsemblExonPDB_map"

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

    # get all unique residues from both exons individually
    #residues1 = set([item for sublist in exons1.values() for item in sublist])
    #residues2 = set([item for sublist in exons2.values() for item in sublist])

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

    residues1_pdb = get_unique_residues(pdb_path1)
    residues2_pdb = get_unique_residues(pdb_path2)
    #print(len(residues1_pdb), len(residues2_pdb))

    if len(residues1_pdb) != len(prot1):
        print("Error: Number of residues in prediction does not match number in pdb 1")
    if len(residues2_pdb) != len(prot2):
        print("Error: Number of residues in prediction does not match number in pdb 2")

    pdb_dict1 = dict(zip(residues1_pdb, prot1))
    pdb_dict2 = dict(zip(residues2_pdb, prot2))

    ex_values1 = dict()
    ex_values2 = dict()

    # we expect each of the residues in exon file must have a value in pdb file
    for ex in exons1:
        ex_values1[ex] = [pdb_dict1[res] for res in exons1[ex]]
    for ex in exons2:
        ex_values2[ex] = [pdb_dict2[res] for res in exons2[ex]]
    #print(sum(len(v) for v in ex_values1.values()), sum(len(v) for v in ex_values2.values()))
    result = {}
    for k1, v1 in ex_values1.items():
        for k2, v2 in ex_values2.items():
            result[(k1, k2)] = np.outer(np.asarray(v1, dtype=object),np.asarray(v2, dtype=object))
            #result[(k1, k2)] = np.add.outer(np.asarray(v1, dtype=object),np.asarray(v2, dtype=object))
   
    
    df = pd.read_csv("../../../data_collection/int_exon_pairs.txt", sep = "\t")
    for k in result:
        pred = np.max(result[k].ravel())
        #try: 
        #    flat_arr = result[k].ravel()
        #    top_5_indices = np.argpartition(flat_arr, -3)[-3:]
        #    top_5 = flat_arr[top_5_indices]
        #    pred = np.mean(top_5)
        #except:
        #    print(result[k].ravel())
        #    pred = np.mean(result[k].ravel())
        if not df[(df['exon1'] == k[0]) & (df['exon2'] == k[1])].empty:
            try: 
                preds.append(pred.item())
            except:
                preds.append(pred)
            true.append(1)
        else: 
            try: 
                preds.append(pred.item())
            except:
                preds.append(pred)
            true.append(0)
#np.save(f"preds_pi_mul_train_{fold+1}.npy", preds)
#np.save(f"true_pi_mul_train_{fold+1}.npy", true)

print(counter)
print(roc_auc_score(true, preds))
#%%
# get positive and negative examples
#preds = np.load(f"preds_pi_mul_train_{fold+1}.npy")
#true = np.load(f"true_pi_mul_train_{fold+1}.npy")
positive_preds = []
negative_preds = []
for pred, label in zip(preds, true):
    if label == 1:
        positive_preds.append(pred)
    else:
        negative_preds.append(pred)

# %%
print(len(positive_preds), len(negative_preds))
# %%
np.save(f"pos_preds_pi_{fold+1}.npy", np.asarray(positive_preds))
np.save(f"neg_preds_pi_{fold+1}.npy", np.asarray(negative_preds))
# %%
