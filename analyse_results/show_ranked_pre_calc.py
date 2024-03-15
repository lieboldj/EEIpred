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

def get_distributions(alpha, pos_dist, neg_dist, background):
    cutoff = np.percentile(background, (1 - alpha) * 100)
    tp = np.sum(np.array(pos_dist) > cutoff)
    fp = np.sum(np.array(neg_dist) > cutoff)
    fn = np.sum(np.array(pos_dist) <= cutoff)
        #tn = np.sum(np.array(neg_dist) <= cutoff)
    return tp, fp, fn#, tn

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

parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-m', '--method', type=str, default="dMaSIF,PInet,GLINTER", help='Methods to test')
parser.add_argument('-p', '--pp', type=str, default="DL", help='Preprocessings to test')
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
#ranked_results = np.zeros((len(thresholds), 9, 3))
roc_aucs = np.zeros((len(thresholds), 9))
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 10), sharex=True)
fig, ax1 = plt.subplots(1)#subplots(1, figsize=(12, 4), sharex=True)
thresholds = [0.01,0.02,0.03,0.04,0.05]
print(thresholds)
ranked_results = np.zeros((len(thresholds), 9, 3, 5))
roc_aucs = np.zeros((len(thresholds), 9))
# get ranked results precision, recall, and f-score
for l, pp in enumerate(pps):
    print(pp)
    print("dMaSIF", "PInet", "GLINTER")
    print("F-score")
    print("Precision")
    print("Recall")

    ranked_results = np.load(f"../results/pre-calc/ranked_results_data_{pp}_perFold.npy")
    
    # 
    print(ranked_results.shape)
    
    #print(ranked_results)
    #print("0,01: ", ranked_results[0])
    #print("0,05: ", ranked_results[4])
    #print("0,1: ", ranked_results[8])
    #print(ranked_results.shape)
    # write results to file
    # rows are methods + pp + dataset and columns are metric (F-score, precision, recall)
    # + threshold (0.01, 0.05, 0.1)
    # write to file
    #list_e= ["dMaSIF", "PInet", "GLINTER"]
    #with open(f"results/plots/ranked_results_data_{pp}.txt", "w") as f:
    #    for k in range(3):
    #        f.write(list_e[k] + "\n")
    #        #f.write(str(list_e[k]) + ",")
    #        for i in [0,1,2,3,4]: # threshold
    #            f.write("Threshold: " + str(thresholds[i]) + "\n")
    #            f.write("CONTACT\n")
    #            for j in [0,1,2]:
    #                for n in range(5):
    #                    f.write("," +str(ranked_results[i, j, k, n]))
    #                #f.write(str(ranked_results[i, j, k]) + ",")
    #            f.write("\n")
    #            f.write("PISA\n")
    #            for j in [3,4,5]:
    #                for n in range(5):
    #                    f.write("," + str(ranked_results[i, j, k, n]))
    #                #f.write(str(ranked_results[i, j, k]) + ",")
    #            f.write("\n")
    #            f.write("EPPIC\n")
    #            for j in [6,7,8]:
    #                for n in range(5):
    #                    f.write("," + str(ranked_results[i, j, k, n]))
    #                #f.write(str(ranked_results[i, j, k]) + ",")
    #            f.write("\n")
    #        f.write("\n")
#

    #print(ranked_results.shape)
    #print(len(thresholds))
    
    ranking = np.zeros((len(thresholds), 3))
    # get the absolute numbers of how often a method is best
    #for i in range(len(thresholds)):
    #    #print(thresholds[i])
    #    for m in range(9):
    #        #print(ranked_results[i, m, :].shape, ranked_results[i, m, :])
    #        if np.argmax(ranked_results[i, m]) == 0:
    #            ranking[i, 0] += 1
    #        elif np.argmax(ranked_results[i, m]) == 1:
    #            ranking[i, 1] += 1
    #        elif np.argmax(ranked_results[i, m]) == 2:
    #            ranking[i, 2] += 1

    for i in range(5):
        for m in range(9):
            for n in range(5):
                #print(ranked_results[i, m, :,n].shape)
                if np.argmax(ranked_results[i, m, :,n]) == 0:
                    #ranked_results[i, m, 0,n] = 1
                    #print("dMaSIF: ", m,n)
                    ranking[i, 0] += 1
                elif np.argmax(ranked_results[i, m, : ,n]) == 1:
                    #ranked_results[i, m, 1,n] = 1
                    ranking[i, 1] += 1
                    #print("PInet: ", m,n)
                elif np.argmax(ranked_results[i, m, :,n]) == 2:
                    #ranked_results[i, m, 2,n] = 1
                    ranking[i, 2] += 1
                    #print("GLINTER: ", m,n)

    font_size = 16
    #plt.figure(figsize=(10, 5))
    x = np.arange(len(thresholds))
    x = ["1%", "2%", "3%", "4%", "5%"]
    width = 0.6  # Adjust the width of the bars as needed
    # Plot the bars with stacked colors
    if l == 2:
        ax3.bar(x, ranking[:, 0], width, label="dMaSIF", color='#0000a7', alpha=0.7)
        ax3.bar(x, ranking[:, 1], width, bottom=ranking[:, 0], label="PInet", color='#008176', alpha=0.7)
        ax3.bar(x, ranking[:, 2], width, bottom=ranking[:, 0] + ranking[:, 1], label="GLINTER", color='#eecc16', alpha=0.7)
        #a31.set_xticks(x, thresholds, fontsize=font_size)
        ax3.set_yticks(range(1,10, 2))# fontsize=font_size)
        ax3.set_title("(c) EEI prediction using PPDL", fontsize=font_size)
        ax3.yaxis.set_tick_params(labelsize=font_size)
        ax3.set_ylim(0, 9)
    if l == 1:
        ax2.bar(x, ranking[:, 0], width, label="dMaSIF", color='#0000a7', alpha=0.7)
        ax2.bar(x, ranking[:, 1], width, bottom=ranking[:, 0], label="PInet", color='#008176', alpha=0.7)
        ax2.bar(x, ranking[:, 2], width, bottom=ranking[:, 0] + ranking[:, 1], label="GLINTER", color='#eecc16', alpha=0.7)
        #ax2.set_xticks(x, thresholds, fontsize=font_size)
        #ax2.set_yticks(range(1,10, 2))
        ax2.set_yticks(range(0,46, 5))
        ax2.set_ylim(0, 9)
        ax2.set_ylabel("Absolute number of top ranks (out of 9)", fontsize=font_size)
        ax2.set_title("(b) EEI prediction using PPMax", fontsize=font_size)
        ax2.yaxis.set_tick_params(labelsize=font_size)
        ax2.set_ylim(0, 45)  
        offset = 3
        for rect in bar1:
            height = rect.get_height()
            if height == 0:
                continue
            if height < offset:
                print(rect.get_height())
                continue
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                height - offset,  # Adjust the vertical position of the text
                f'{int(rect.get_height())}',
                ha='center',
                va='bottom',
                fontsize=font_size,
                color='white'
            )
        for i, rect in enumerate(bar2):
            height = rect.get_height()# + ranking[:, 0]
            if height == 0:
                continue
            height += ranking[i, 0]
            if height < offset:
                print(rect.get_height())
                continue
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                height - offset,  # Adjust the vertical position of the text
                f'{int(rect.get_height())}',
                ha='center',
                va='bottom',
                fontsize=font_size
            )

        for i, rect in enumerate(bar3):
            height = rect.get_height()# + ranking[:, 0] + ranking[:, 1]
            if height == 0:
                continue
            height += ranking[i, 0] + ranking[i, 1]
            if height < offset:
                print(rect.get_height())
                continue
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                height - offset,  # Adjust the vertical position of the text
                f'{int(rect.get_height())}',
                ha='center',
                va='bottom',
                fontsize=font_size
            )
        # set xticks to 1%...5%
        #ax2.set_xticks(x)
        #ax2.set_yticks(fontsize=font_size)
    if l == 0:
        bar1 = ax1.bar(x, ranking[:, 0], width, label="dMaSIF", color='#0000a7', alpha=0.7)
        bar2 = ax1.bar(x, ranking[:, 1], width, bottom=ranking[:, 0], label="PInet", color='#008176', alpha=0.7)
        bar3 = ax1.bar(x, ranking[:, 2], width, bottom=ranking[:, 0] + ranking[:, 1], label="GLINTER", color='#eecc16', alpha=0.7)
        #a13.set_xticks(x, thresholds, fontsize=font_size)
        ax1.set_yticks(range(0,46, 5))
        #ax1.set_yticks([]*9)
        ax1.yaxis.set_tick_params(labelsize=font_size)
        ax1.set_ylim(0, 45)  
        offset = 3
        for rect in bar1:
            height = rect.get_height()
            if height == 0:
                continue
            if height < offset:
                print(rect.get_height())
                continue
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                height - offset,  # Adjust the vertical position of the text
                f'{int(rect.get_height())}',
                ha='center',
                va='bottom',
                fontsize=font_size,
                color='white'
            )
        for i, rect in enumerate(bar2):
            height = rect.get_height()# + ranking[:, 0]
            if height == 0:
                continue
            if height < offset:
                print(rect.get_height())
                continue
            height += ranking[i, 0]
            
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                height - offset,  # Adjust the vertical position of the text
                f'{int(rect.get_height())}',
                ha='center',
                va='bottom',
                fontsize=font_size
            )

        for i, rect in enumerate(bar3):
            height = rect.get_height()# + ranking[:, 0] + ranking[:, 1]
            if height == 0:
                continue
            if height < offset:
                print(rect.get_height())
                continue
            height += ranking[i, 0] + ranking[i, 1]
            
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                height - offset,  # Adjust the vertical position of the text
                f'{int(rect.get_height())}',
                ha='center',
                va='bottom',
                fontsize=font_size
            )
       
        #a13.set_yticks(fontsize=font_size)
        #ax1.set_title("(a) RRI prediction", fontsize=font_size)

plt.xticks(x, fontsize=font_size)
plt.xlabel("False Discovery Rate (FDR)", fontsize=font_size, labelpad=15)
plt.ylabel("Number of performance tests", fontsize=font_size, labelpad=15)
if pp=="Max":
    plt.legend(loc='lower left', ncol=1, fontsize=font_size-2, bbox_to_anchor=(1, 0.15))
else:
    plt.legend(loc='lower right', ncol=3, fontsize=font_size-2)

# set grid for x axes only
#plt.grid(axis='y')
plt.savefig(f"../results/plots/ranked/ranking_45_0225{pp}.png", dpi=600, bbox_inches='tight')



