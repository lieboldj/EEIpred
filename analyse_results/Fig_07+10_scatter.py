

#%%
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

#%%
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-mh', '--method', type=str, default="dMaSIF,PInet,GLINTER,ProteinMAE", help='Methods to test')
parser.add_argument('-p', '--pp', type=str, default="AA,DL", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CLUST_CONTACT,CLUST_PISA,CLUST_EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AU-ROC')
args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")

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

# check if metric_string is in the list of metrics
metrics = ["AUROC", "AUPRC", "Fscore", "Precision", "Recall", "MCC"]
metric_legend = ["AU-ROC", "AU-PRC", "F-score", "Precision", "Recall", "MCC"]
all_methods = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
all_pps = ["AA", "Max", "DL"]
all_datasets = ["CLUST_CONTACT","CLUST_PISA", "CLUST_EPPIC"]
plt.figure(figsize=(6,6))

metric_marker = ["o", "p", "X", "*", "^", "s"]
method_color = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
dataset_density = [0.2,0.5,1]

for e, metric_string in enumerate(metrics):
    print(e, metric_string)
    metric_check = True
    data = dict()        
    for alphas in [[0.01],[0.02],[0.03],[0.04],[0.05]]: #
        if "CLUST" in datasets[0]:
            data = np.load(f"../results/plots/all_clust{alphas[0]}.npy", allow_pickle=True).item()  
        else:
            data = np.load(f"../results/plots/all_{alphas[0]}.npy", allow_pickle=True).item()
        #data = np.load(f"../results/plots/all_{alphas[0]}.npy", allow_pickle=True).item() 

        data_Max = dict()
        data_DLs = dict()
        print(alphas)

        for key in data.keys():
            if pps[1] in key:
                data_DLs[key] = data[key]
            elif pps[0] in key:
                data_Max[key] = data[key]
        # create scatter plot for each combination of DL and DLs pp
        # the x axis is DLs and the y axis is DL
        # for each combination of method and dataset choose a color
        if metric_check:
            for m, method in enumerate(methods):
                for dataset in datasets:
                        label_plot_DL = f"{method} {dataset} - {pps[0]} - {metric_string}"
                        label_plot_DLs = f"{method} {dataset} - {pps[1]} - {metric_string}"
                        plt.scatter(np.mean(data_Max[label_plot_DL]), np.mean(data_DLs[label_plot_DLs]), marker=metric_marker[e], color=method_color[m], alpha=dataset_density[datasets.index(dataset)])
                    #plt.scatter(data_Max[label_plot_DL], data_DLs[label_plot_DLs], label=f"{method}-{dataset}: {y_label}", marker=metric_marker[e], color=method_color[m], alpha=dataset_density[datasets.index(dataset)])
        if metric_string == "AUPRC" or metric_string == "AUROC":
            metric_check = False
plt.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=0.5)
if pps[0] == "Max":
    plt.xlim(0,0.85)
    plt.ylim(0,0.85)
else:
    plt.xlim(0,1)
    plt.ylim(0,1)
fontsize = 16
if pps[0] == "AA":
    #plt.title("RRI vs PPDL", fontsize=fontsize)
    plt.xlabel("RRI prediction performance", fontsize=fontsize, labelpad=15)
else:
    plt.xlabel("EEI prediction performance (using PPMax)", fontsize=fontsize, labelpad=15)
    #plt.title("PPMax vs PPMaxDL", fontsize=fontsize)
plt.ylabel("EEI prediction performance (using PPDL)", fontsize=fontsize, labelpad=15)
#plt.ylabel("RRI prediction performance - 8A", fontsize=fontsize, labelpad=15)

#leg = plt.legend(ncol=3, loc='upper left', bbox_to_anchor=(1.15, 1.15))
#place legend outside of plot
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
legend_elements = []
f = lambda m,c,a: plt.plot([],[],marker=m, color=c, alpha=a, ls="none")[0]

handles = [f("s", method_color[i],1) for i in range(len(methods))]
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))
legend_elements += handles

# Horizontal line between methods and dataset_names
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))

#handles += [f("s", "k", dataset_density[i]) for i in range(3)]
legend_elements += [f("s", "k", dataset_density[i]) for i in range(len(datasets))]
legend_elements.append(plt.Line2D([0], [1], color='white', linestyle='-', linewidth=1))
#handles += [f(metric_marker[i], "k",1) for i in range(5)]
legend_elements += [f(metric_marker[i], "k",1) for i in range(len(metrics))]

dataset_names = ["$D_{Con}$", "$D_{Engy}$", "$D_{Evol}$"]
labels = ["$\mathbf{PPIIP}$\n$\mathbf{method}$"] + methods + ["$\mathbf{Dataset}$"] + dataset_names+ ["$\mathbf{Performance}$\n$\mathbf{measure}$"]+metric_legend
#plt.grid()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

leg = plt.legend(legend_elements, labels, loc='lower left', fontsize=fontsize-3, bbox_to_anchor=(1.02, -0.03))
plt.savefig(f"{pps[0]}_{pps[1]}_CLUST.png", dpi=600, \
                bbox_extra_artists=(leg,), bbox_inches='tight')

plt.close()


