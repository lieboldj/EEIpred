import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# Given functions
def precision(TP, FP):
    return TP / (TP + FP)
def recall(TP, FN):
    return TP / (TP + FN)
def f_score(TP, FP, FN):
    return 2 * precision(TP, FP) * recall(TP, FN) / (precision(TP, FP) + recall(TP, FN))

def get_results(alphas, pos_dist, neg_dist, background):
    prec = list()
    rec = list()
    fscores = list()

    alpha = alphas
    cutoff = np.percentile(background, (1 - alpha) * 100)
    tp = np.sum(np.array(pos_dist) > cutoff)
    fp = np.sum(np.array(neg_dist) > cutoff)
    fn = np.sum(np.array(pos_dist) <= cutoff)
    
    prec.append(precision(tp, fp))
    rec.append(recall(tp, fn))
    fscores.append(f_score(tp, fp, fn))
    return prec, rec, fscores

def get_distributions(alphas, pos_dist, neg_dist, background):
    for alpha in alphas:
        cutoff = np.percentile(background, (1 - alpha) * 100)
        tp = np.sum(np.array(pos_dist) > cutoff)
        fp = np.sum(np.array(neg_dist) > cutoff)
        fn = np.sum(np.array(pos_dist) <= cutoff)
        tn = np.sum(np.array(neg_dist) <= cutoff)
    return tp, fp, fn, tn
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-m', '--method', type=str, default="dMaSIF,PInet,GLINTER", help='Methods to test')
parser.add_argument('-p', '--pp', type=str, default="AA,DL", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CONTACT,PISA,EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=bool, default=True, help='Sampling on or off')
parser.add_argument('-a', '--auroc', type=bool, default=False, help='Set to True if you want to know AUROC and AUPRC')
args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
sampling_on = args.sampling
auroc_on = args.auroc


thresholds = [i / 100 for i in range(1, 6)]
ranked_results = np.zeros((len(thresholds), 9, 3))

# get ranked results precision, recall, and f-score
for l, pp in enumerate(pps):
    thresholds = [i / 100 for i in range(1, 6)]
    ranked_results = np.load(f"../results/pre-calc/ranked_results_data_{pp}_perFold.npy")
    print(ranked_results.shape)
    #ranked_results_std = np.load(f"results/plots/ranked_results_std_data_{pp}_perFold.npy")
    # get ranked_results_mean and ranked_results_std
    ranked_results_mean = np.zeros((len(thresholds), 9, 3))
    ranked_results_std = np.zeros((len(thresholds), 9, 3))
    for i, threshold in enumerate(thresholds):
        for j in range(9):
            for k in range(3):
                ranked_results_mean[i, j, k] = np.mean(ranked_results[i, j, k])
                ranked_results_std[i, j, k] = np.std(ranked_results[i, j, k])
    
    # plot for each metric precision, recall, and f-score the line plot with alpha on x-axis and metric on y-axis
    all_colors = ['lightblue', 'lightskyblue', 'royalblue', 'palegreen', 'limegreen',\
                   'seagreen', 'lightcoral', 'firebrick', 'maroon']
    colors = ["#0000a7", "#008176", "#eecc16"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 10), sharex=True)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    fontsize = 19

    ### barplot + error
    idx = 3
    for i in range(3):
        if i == 0:
            thresholds = [i - 0.0025 for i in thresholds]    
        else:
            # ,ove thresholds a bit to the right
            thresholds = [i + 0.0025 for i in thresholds]
        ax1.bar(thresholds, ranked_results_mean[:, idx, i], yerr = ranked_results_std[:, idx, i], width=0.0025, label=f"{methods[i]}", color=colors[i], alpha=0.7)

    ax1.set_ylabel("F-score", fontsize=fontsize, labelpad=15)
    if pp == "AA":
        ax1.set_ylim([0, 0.06])
        ax1.set_yticklabels([0.0,0.01,0.02,0.03,0.04,0.05,0.06], fontsize=fontsize) # for y-axis 0 to 0.5
    else:
        ax1.set_ylim([0, 0.4])
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        for n, label in enumerate(ax1.yaxis.get_ticklabels()):
            if n % 2 != 0:
                label.set_visible(False)

        ax1.set_yticklabels([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4], fontsize=fontsize) # for y-axis 0 to 0.5

    ax1.set_xlabel("False Discovery Rate (FDR)", fontsize=fontsize, labelpad=15)
    ax1.set_xlim([0.005, 0.055])
    ax1.set_xticklabels([0, "1%", "2%", "3%", "4%", "5%"], fontsize=fontsize)
                         
    plt.legend(fontsize=fontsize, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"../results/plots/precision_thres/results_thes{pp}_pre_{idx}_0224.png", dpi=300)
    plt.close()
