#%%
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glob
#%%

eppic = pd.read_csv("../../../data_collection/EPPIC_EEIN_positive.txt", sep='\t')
pisa = pd.read_csv("../../../data_collection/PISA_EEIN_0.5_positives.txt", sep='\t')
#get all pairs of exon1 and exon2 from eppic
eppic_pairs = set()
for i in range(len(pisa)):
    try:
        eppic_pairs.add(pisa.iloc[i]['exon1'] + "_" + pisa.iloc[i]['exon2'])
    except:
        print(i)
        continue
print(len(eppic_pairs))

#%%

dataset = "PISA"
mode = "test"
# %%
for i in range(1,6):
    pos = []
    neg = []
    files = glob.glob(f"../../results/{dataset}/fold_{i}/{mode}/" + '*.npy')
    files.extend(glob.glob(f"../../results/{dataset}/fold_{i}/{mode}/large/" + '*.npy'))
    for file in tqdm(files):
        pair = np.load(file)
        exons = file.split('/')[-1].split('_')
        if exons[-2] + "_" + exons[-1][:-4] in eppic_pairs or \
            exons[-1][:-4] + "_" + exons[-2] in eppic_pairs:
            pos.append(np.max(pair))
        else:
            neg.append(np.max(pair))

    np.save(f"back-fore/{dataset}/pos_{i}.npy", pos)
    np.save(f"back-fore/{dataset}/neg_{i}.npy", neg)


# %%
from sklearn.metrics import roc_curve, roc_auc_score
dataset = "EPPIC"
for i in range(1,6):
    neg = np.load(f"back-fore/{dataset}/neg_{i}.npy")
    pos = np.load(f"back-fore/{dataset}/pos_{i}.npy")
    back = np.load(f"back-fore/{dataset}/back_{i}.npy")
    true = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
    pred = np.concatenate((pos, neg))
    fpr, tpr, thresholds = roc_curve(true, pred)
    auc_score = roc_auc_score(true, pred)

# %%
