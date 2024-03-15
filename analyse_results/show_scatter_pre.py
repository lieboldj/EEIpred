

#%%
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

#%%
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-mh', '--method', type=str, default="dMaSIF,PInet,GLINTER", help='Methods to test')
parser.add_argument('-p', '--pp', type=str, default="AA,DL", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CONTACT,PISA,EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AU-ROC')
parser.add_argument('-mr','--metric', type=str, default="F-score", help='Metric to plot, choose between AU-PRC, AU-ROC, F-score, Precision, Recall, for Precision and Recall you need to provide alpha')
args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
metric_string = args.metric
if args.sampling == 1:
    sampling_on = True
else:
    sampling_on = False


# get statistical evaluation with TP, FP, FN, TN
def precision(TP, FP):
    return TP / (TP + FP)
def recall(TP, FN):
    return TP / (TP + FN)
def specificity(TN, FP):
    return TN / (TN + FP)
def false_positive_rate(FP, TN):
    return FP / (FP + TN)
def false_negative_rate(FN, TP):
    return FN / (FN + TP)
def true_positive_rate(TP, FN):
    return TP / (TP + FN)
def f_score(TP, FP, FN):
    return 2 * precision(TP, FP) * recall(TP, FN) / (precision(TP, FP) + recall(TP, FN))
def mcc(TP, FP, TN, FN):
    return (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# %%
def get_results(alphas, pos_dist, neg_dist, background):
    prec = list()
    rec = list()
    fscores = list()
    mccs = list()
    cutoffs = list()
    warn = False

    for alpha in alphas:
        cutoff = np.percentile(background, (1 - alpha) * 100)
        tp = np.sum(np.array(pos_dist) > cutoff)
        fp = np.sum(np.array(neg_dist) > cutoff)
        fn = np.sum(np.array(pos_dist) < cutoff)
        tn = np.sum(np.array(neg_dist) < cutoff)
        if tp == 0 and fp == 0:
            warn = True
            prec.append(np.nan)
            rec.append(np.nan)
            fscores.append(np.nan)
        else:
            prec.append(precision(tp, fp))
            rec.append(recall(tp, fn))
            fscores.append(f_score(tp, fp, fn))
        #mccs.append(mcc(tp, fp, tn, fn))
        cutoffs.append(cutoff)
    return prec, rec, fscores, warn 
# check if metric_string is in the list of metrics
metrics = ["AU-ROC", "AU-PRC", "F-score", "Precision", "Recall"]
if metric_string not in metrics:
    print("Please choose a metric from the following list: ", metrics)
    exit()
all_methods = ["dMaSIF", "PInet", "GLINTER"]
all_pps = ["AA", "Max", "DL"]
all_datasets = ["CONTACT","PISA", "EPPIC"]
plt.figure(figsize=(6,6))

metric_marker = ["o", "p", "X", "*", "^", "X"]
method_color = ["#0000a7", "#008176", "#eecc16"]
dataset_density = [0.2,0.5,1]

for e, metric_string in enumerate(metrics):
    metric_check = True
    data = dict()        
    for alphas in [[0.01],[0.02],[0.03],[0.04],[0.05]]:
        data = np.load(f"../results/pre-calc/{metric_string}_all_{alphas[0]}_AADLS.npy", allow_pickle=True).item()
        if metric_string == "Recall":
            print("threshold", alphas[0])

        data_Max = dict()
        data_DLs = dict()
        print(alphas)

        for key in data.keys():
            if "DLs" in key:
                data_DLs[key] = data[key]
            else:
                data_Max[key] = data[key]

        # create scatter plot for each combination of DL and DLs pp
        # the x axis is DLs and the y axis is DL
        # for each combination of method and dataset choose a color
        if metric_check:
            for m, method in enumerate(methods):
                for dataset in datasets:
                    label_plot_DL = f"{method} {dataset} - AA"
                    label_plot_DLs = f"{method} {dataset} - DLs"
                    plt.scatter(np.mean(data_Max[label_plot_DL]), np.mean(data_DLs[label_plot_DLs]), marker=metric_marker[e], color=method_color[m], alpha=dataset_density[datasets.index(dataset)])
                    #plt.scatter(data_Max[label_plot_DL], data_DLs[label_plot_DLs], label=f"{method}-{dataset}: {y_label}", marker=metric_marker[e], color=method_color[m], alpha=dataset_density[datasets.index(dataset)])
        if metric_string == "AU-PRC" or metric_string == "AU-ROC":
            metric_check = False
plt.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=0.5)
plt.xlim(0,1)
plt.ylim(0,1)
fontsize = 16

plt.xlabel("RRI prediction performance", fontsize=fontsize, labelpad=15)
#plt.xlabel("EEI prediction performance (using PPMax)", fontsize=fontsize, labelpad=15)
plt.ylabel("EEI prediction performance (using PPDL)", fontsize=fontsize, labelpad=15)

#leg = plt.legend(ncol=3, loc='upper left', bbox_to_anchor=(1.15, 1.15))
#place legend outside of plot
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
legend_elements = []
f = lambda m,c,a: plt.plot([],[],marker=m, color=c, alpha=a, ls="none")[0]

handles = [f("s", method_color[i],1) for i in range(3)]
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))
legend_elements += handles

# Horizontal line between methods and dataset_names
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))

#handles += [f("s", "k", dataset_density[i]) for i in range(3)]
legend_elements += [f("s", "k", dataset_density[i]) for i in range(3)]
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))
#handles += [f(metric_marker[i], "k",1) for i in range(5)]
legend_elements += [f(metric_marker[i], "k",1) for i in range(5)]

dataset_names = ["$D_{Con}$", "$D_{Engy}$", "$D_{Evol}$"]
labels = ["$\mathbf{PPIIP}$\n$\mathbf{method}$"] +methods + ["$\mathbf{Dataset}$"] + dataset_names+ ["$\mathbf{Performance}$\n$\mathbf{measure}$"]+metrics
#plt.grid()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

leg = plt.legend(legend_elements, labels, loc='lower left', fontsize=fontsize-3, bbox_to_anchor=(1.02, 0.05))
plt.savefig(f"../results/plots/scatter/AAaVSDL_260224.png", dpi=600, \
                bbox_extra_artists=(leg,), bbox_inches='tight')
#if sampling_on:
#    plt.savefig(f"results/plots/DLsVSDL_all_sampling_{alphas[0]}.png", dpi=300)
#else:
#    plt.savefig(f"results/plots/scatter/AAaVSDL_all_c.png", dpi=600, \
#                bbox_extra_artists=(leg,), bbox_inches='tight')
plt.close()


