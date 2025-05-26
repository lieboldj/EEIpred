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

parser = argparse.ArgumentParser(description='Create boxplots for BioGRID PPIs.')
parser.add_argument('-m', '--method', type=str, default='dMaSIF', help='Method to evaluate')

args = parser.parse_args()

datasets = ["CLUST_CONTACT"]#, "CLUST_PISA", "CLUST_EPPIC"]
method = args.method
only_lower100 = True # only use small exon pairs, there we got predictions

for fold in range(1,2):
    if fold == 1:
        # load the background model which we used as the pre-trained model
        test_neg = np.load(f"../results/ProteinMAE_DL/{datasets[0]}_train_neg_fold{fold}.npy")
    else:
        test_neg = np.concatenate((test_neg, np.load(f"../results/ProteinMAE_DL/{datasets[0]}_test_neg_fold{fold}.npy")))
biogrid_scores = []

biogrid_pkl = "../results/ProteinMAE_DL/BioGRID_fold1_results.pkl"
with open(biogrid_pkl, "rb") as f:
    biogrid_dict = pickle.load(f)
print(len(biogrid_dict))
# print(len(test_pos))
print(len(test_neg)) 

biogrid_scores = []
# get all values from biogrid_scores
for key, value in biogrid_dict.items():
    biogrid_scores.append(value)

# make boxplot for biogrid_scores, test_pos and test_neg
top_5_percentile = np.percentile(test_neg, 95)
# how many of test_neg are above top_5_percentile
print("Number of test_neg above top_5_percentile:", len([x for x in test_neg if x > top_5_percentile]))
max_score_neg = np.max(test_neg)

# Create the figure
plt.figure(figsize=(6, 6))
box = plt.boxplot(
    [biogrid_scores, test_neg],
    #patch_artist=True,
    tick_labels=["BioGRID", "Background"],
    widths=0.7,
    medianprops=dict(color='blue', linewidth=1),
    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, linestyle='none')
)

# Add reference lines
plt.axhline(top_5_percentile, color='orange', linestyle='--', linewidth=1.5, label="95th percentile of background")
plt.axhline(max_score_neg, color='red', linestyle='--', linewidth=1.5, label="Maximum background score")
plt.axhline(1, color='black', linestyle='--', linewidth=1.5, label="Maximum possible score")
# Annotate lines (optional: adjust y-offset as needed)
plt.text(1.05, top_5_percentile + 0.01, '1% False discovery rate threshold', color='brown', fontsize=10)
plt.text(1.05, max_score_neg + 0.035, 'Scores > Maximum score of \nthe background distribution', color='red', fontsize=10)
plt.text(1.05, 1 + 0.01, 'Maximum possible score for an exon pair', color='black', fontsize=10)

#cover the area between the two lines
plt.fill_betweenx([max_score_neg, 1], 0.5, 2.5, color='red', alpha=0.2, label="False discovery rate area")
#plt.fill_betweenx([top_5_percentile, max_score_neg], 0.5, 2.5, color='orange', alpha=0.2, label="False discovery rate area")

# Formatting
#plt.title("EEI prediction scores for BioGRID vs. background", fontsize=14, weight='bold')
plt.ylabel("Prediction scores for exon pairs using ProteinMAE + PPDL", fontsize=12, labelpad=10)
plt.xlabel("Evaluation set", fontsize=12, labelpad=10)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, 1.05)
#plt.grid(axis='y', linestyle=':', alpha=0.5)
#plt.legend(loc='upper left', fontsize=10)

# Save the plot
plt.tight_layout()
plt.savefig("plots/biogrid_boxplot_DL_2.png", dpi=600)
plt.close()
print(np.max(biogrid_scores))
# print the score of biogrid_scores which is the closest above the max_score_neg
closest_above = min((x for x in biogrid_scores if x > top_5_percentile), default=None)
print(max_score_neg)
print(top_5_percentile)
print("Closest score above max_score_neg:", closest_above)

biogrid_dict = {k: v for k, v in sorted(biogrid_dict.items(), key=lambda item: item[1], reverse=True)}

# keep only pairs with value > max_score_neg
biogrid_dict = {k: v for k, v in biogrid_dict.items() if v > top_5_percentile}
print(len(biogrid_dict))
# get all keys from biogrid_dict
biogrid_keys = []
for key, value in biogrid_dict.items():
    # sort key[0] and key[1] by name
    proteins = tuple(sorted([key[0], key[1]]))
    biogrid_keys.append(proteins)
# make a set from biogrid_keys
print(len(set(biogrid_keys)))

#print total number in biogrid scores, then number of pair above the 95% line
print("Total number of BioGRID scores:", len(biogrid_scores))
print("Number of BioGRID scores above 95% line:", len([x for x in biogrid_scores if x > np.percentile(test_neg, 95)]))
print("Number of BioGRID scores above max line:", len([x for x in biogrid_scores if x > np.max(test_neg)]))

# check for highest scores in 
biogrid_pkl = "../results/ProteinMAE_DL/BioGRID_fold1_results.pkl"

with open(biogrid_pkl, "rb") as f:
    af_biogrid = pickle.load(f)
# sorted biogrid_dict save in .tsv
# order biogrid_dict by value
biogrid_dict = {k: v for k, v in sorted(af_biogrid.items(), key=lambda item: item[1], reverse=True)}
# save to tsv
max_value = np.max(test_neg)
max_bool = False

with open("biogrid_scores.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["Protein1", "Exon1", "Protein2", "Exon2", "Score"])
    for key, value in biogrid_dict.items():
        writer.writerow([key[0], key[2], key[1], key[3], value])
