

#%%
import os
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

#%%
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-mh', '--method', type=str, default="dMaSIF,PInet,GLINTER,ProteinMAE", help='Methods to test')#dMaSIF,PInet,GLINTER,
parser.add_argument('-p', '--pp', type=str, default="AA", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CLUST_CONTACT,CLUST_PISA,CLUST_EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AUROC')
parser.add_argument('-mr','--metric', type=str, default="AUPRC", help='Metric to plot, choose between AUPRC, AUROC, Fscore, Precision, Recall, for Precision and Recall you need to provide alpha')
parser.add_argument('-a','--alpha', type=float, default=0.05, help='Threshold for precision and recall')
parser.add_argument('-c','--conf_matrix', type=int, default=0, help='Confusion matrix')
parser.add_argument('-csv','--csv', type=int, default=0, help='Create csv file with all results')

args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
metric_string = args.metric
alphas = [args.alpha]
conf_matrix = args.conf_matrix
create_csv = args.csv
if args.sampling == 1:
    sampling_on = True
else:
    sampling_on = False

# check if metric_string is in the list of metrics
metrics = ["AUPRC", "AUROC", "Fscore", "Precision", "Recall", "MCC"]


all_methods = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
all_pps = ["AA", "Max", "DL"]
all_datasets = ["CLUST_CONTACT", "CLUST_EPPIC","CLUST_PISA"]#, "EPPIC"]

dict_groups = methods

#plt.figure(figsize=(6,6))
data = dict()   
use_all = True 
fig, axes = plt.subplots(len(methods), len(datasets), figsize=(4 * len(datasets), 4*len(methods)))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# increase font size
font_size = 12
plt.rcParams.update({'font.size': font_size})

pp = pps[0]
for i, method in enumerate(methods):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]
        scores = []

        for fold in range(1, 6):
            if "CLUST" in dataset:
                pos_scores = np.load(f"../results/{method}_{pp}/{dataset}_test_pos_fold{fold}.npy")
                neg_scores = np.load(f"../results/{method}_{pp}/{dataset}_test_neg_fold{fold}.npy")
            else:   
                pos_scores = np.load(f"../results/{method}_{pp}/{dataset}_pos_fold{fold}.npy")
                neg_scores = np.load(f"../results/{method}_{pp}/{dataset}_neg_fold{fold}.npy")

            y_true = np.concatenate((np.ones(len(pos_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)))
            y_scores = np.concatenate((pos_scores, neg_scores))

            if metric_string == "AUPRC":
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                score = average_precision_score(y_true, y_scores)
                ax.plot(recall, precision, label=f"Fold {fold} (AUPRC={score:.2f})")
            elif metric_string == "AUROC":
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                score = roc_auc_score(y_true, y_scores)
                ax.plot(fpr, tpr, label=f"Fold {fold} (AUROC={score:.2f})")
            elif metric_string in ["Precision", "Recall", "Fscore"]:
                y_pred = [1 if s >= args.alpha else 0 for s in y_scores]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                if metric_string == "Precision":
                    score = tp / (tp + fp) if (tp + fp) > 0 else 0
                elif metric_string == "Recall":
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0
                elif metric_string == "Fscore":
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                ax.bar(fold, score, label=f"Fold {fold}")
            elif metric_string == "MCC":
                y_pred = [1 if s >= args.alpha else 0 for s in y_scores]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                numerator = (tp * tn) - (fp * fn)
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                score = numerator / denominator if denominator != 0 else 0
                ax.bar(fold, score, label=f"Fold {fold}")

        # Only label y-axis on leftmost column
        #if j == 0:
        #    ax.set_ylabel(metric_string, fontsize=font_size)

        # Only label x-axis if not AUPRC/AUROC
        if metric_string not in ["AUPRC", "AUROC"]:
            ax.set_xlabel("Fold")

        # Add method name as y-axis label for the row (centered)
        if j == 0:
            ax.annotate(method, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 15, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)

        # Add dataset name as column title
        if i == 0:
            if "CONTACT" in dataset:
                dat_set = "$D_{Con}$"
            elif "PISA" in dataset:
                dat_set = "$D_{Engy}$"
            else:
                dat_set = "$D_{Evol}$"
            ax.set_title(dat_set, fontsize=12)
        
        # set font size of x and y ticks to font_size
        ax.tick_params(axis='both', labelsize=font_size)


        
        if metric_string == "AUROC":
            ax.set_xlabel("False Positive Rate", fontsize=font_size)
            ax.set_ylabel("True Positive Rate", fontsize=font_size)
            ax.legend(fontsize=font_size-2, loc='lower right')
        elif metric_string == "AUPRC":
            ax.set_xlabel("Recall", fontsize=font_size)
            ax.set_ylabel("Precision", fontsize=font_size)
            ax.legend(fontsize=font_size-2, loc='upper right')

plt.suptitle(f"Precision-Recall curves across folds for each method and dataset for {pp} predictions", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"CLUST_{metric_string}_{pp}_folds.png")
