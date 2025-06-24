import numpy as np
import glob
import csv
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
import argparse
import pandas as pd

only_lower100 = True # only use small exon pairs, there we got predictions

test_neg = np.load(f"../results/ProteinMAE_DL/CLUST_PISA_train_neg_fold5.npy")
test_pos = np.load(f"../results/ProteinMAE_DL/CLUST_PISA_test_pos_fold5.npy")
biogrid_scores = []

biogrid_pkl = "../results/ProteinMAE_DL/BioGRID_fold1_results.pkl"
with open(biogrid_pkl, "rb") as f:
    biogrid_dict = pickle.load(f)
print(np.max(test_neg))

biogrid_scores = []
# get all values from biogrid_scores
for key, value in biogrid_dict.items():
    biogrid_scores.append(value)

# make boxplot for biogrid_scores, test_pos and test_neg
top_5_percentile = np.percentile(test_neg, 99)
# how many of test_neg are above top_5_percentile
print("Number of test_neg above top_5_percentile:", len([x for x in test_neg if x > top_5_percentile]))
max_score_neg = np.max(test_neg)

# Create the figure
plt.figure(figsize=(6, 6))
box = plt.boxplot(
    [biogrid_scores, test_neg],
    #patch_artist=True,
    tick_labels=["BioGRID\n(all inter-protein exon pairs\nin our curated set of PPIs)", "Background\n (inter-protein non-interacting \nexon pairs, training set,\n fold 5, $D_{Engy}$)"],
    widths=0.7,
    medianprops=dict(color='blue', linewidth=1),
    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, linestyle='none')
)
print(np.median(biogrid_scores))

# Add reference lines
plt.axhline(top_5_percentile, color='brown', linestyle='--', linewidth=1.5, label="99th percentile of background")
plt.axhline(max_score_neg, color='red', linestyle='--', linewidth=1.5, label="Maximum background score")
plt.axhline(1, color='black', linestyle='--', linewidth=1.5, label="Maximum possible score")
# Annotate lines (optional: adjust y-offset as needed)
plt.text(1.02, top_5_percentile + 0.01, '1% False discovery rate threshold', color='brown', fontsize=10)
plt.text(1.02, max_score_neg + 0.015, 'Scores > Maximum score of \nthe background distribution', color='red', fontsize=10)
plt.text(1.02, 1 + 0.01, 'Maximum possible score for an exon pair', color='black', fontsize=10)

#cover the area between the two lines
plt.fill_betweenx([max_score_neg, 1], 0.5, 2.5, color='red', alpha=0.2, label="False discovery rate area")
#plt.fill_betweenx([top_5_percentile, max_score_neg], 0.5, 2.5, color='orange', alpha=0.2, label="False discovery rate area")

# Formatting
#plt.title("EEI prediction scores for BioGRID vs. background", fontsize=14, weight='bold')
plt.ylabel("Prediction scores for exon pairs using ProteinMAE + PPDL", fontsize=12, labelpad=10)
plt.xlabel("Dataset", fontsize=12, labelpad=10)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, 1.05)
#plt.grid(axis='y', linestyle=':', alpha=0.5)
#plt.legend(loc='upper left', fontsize=10)

# Save the plot
plt.tight_layout()
plt.savefig("plots/biogrid_boxplot_DL_3.png", dpi=600)
plt.close()
print(np.max(biogrid_scores))
# print the score of biogrid_scores which is the closest above the max_score_neg
closest_above = min((x for x in biogrid_scores if x > top_5_percentile), default=None)
print(max_score_neg)
print(top_5_percentile)
print("Closest score above top 5 percentile:", closest_above)
# clostest above maximu; 
# print the score of biogrid_scores which is the closest below the max_score_neg
print("check max score neg: ", max_score_neg)
closest_below = min((x for x in biogrid_scores if x > max_score_neg), default=None)

print("Closest score above max_score_neg:", closest_below)
print("number of biogrid scores:", len([x for x in biogrid_scores if x > max_score_neg]))

biogrid_pkl = "../results/ProteinMAE_DL/BioGRID_fold1_results.pkl"
with open(biogrid_pkl, "rb") as f:
    biogrid_dict = pickle.load(f)
biogrid_dict = {k: v for k, v in sorted(biogrid_dict.items(), key=lambda item: item[1], reverse=True)}

# keep only pairs with value > max_score_neg
biogrid_dict = {k: v for k, v in biogrid_dict.items() if v > top_5_percentile}
print("all rows in this categrory: ", len(biogrid_dict))
# get all keys from biogrid_dict
biogrid_keys = []
biogrid_singletons = []
for key, value in biogrid_dict.items():
    # sort key[0] and key[1] by name
    proteins = tuple(sorted([key[0], key[1]]))
    biogrid_singletons.append(key[0])
    biogrid_singletons.append(key[1])
    biogrid_keys.append(proteins)
# make a set from biogrid_keys
print("number of protein pairs in these rows ", len(set(biogrid_keys)))
print("number of single proteins: ", len(set(biogrid_singletons)))

# # check for highest scores in 
# biogrid_pkl = "../results/ProteinMAE_DL/BioGRID_fold1_results.pkl"

# with open(biogrid_pkl, "rb") as f:
#     af_biogrid = pickle.load(f)
# # get the highest values in af_biogrid
# highest_values = []
# counter = 0

# print(np.max(test_pos))
# for key, value in af_biogrid.items():
#     counter += 1
#     if value > np.max(test_pos) and key[0] != key[1]:
#         highest_values.append((key, value))

# print(len(highest_values))
# print(highest_values)

# # sort highest_values by value
# highest_values.sort(key=lambda x: x[1], reverse=True)
# print(highest_values)

# # sorted biogrid_dict save in .tsv
# # order biogrid_dict by value
# biogrid_dict = {k: v for k, v in sorted(biogrid_dict.items(), key=lambda item: item[1], reverse=True)}
# # save to tsv
# with open("biogrid_scores_pisa.tsv", "w") as f:
#     writer = csv.writer(f, delimiter="\t")
#     writer.writerow(["Protein1", "Exon1", "Protein2", "Exon2", "Score"])
#     for key, value in biogrid_dict.items():
#         writer.writerow([key[0], key[2], key[1], key[3], value])
