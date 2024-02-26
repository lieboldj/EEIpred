

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
parser.add_argument('-p', '--pp', type=str, default="AA", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CONTACT,PISA,EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AUROC')
parser.add_argument('-mr','--metric', type=str, default="AUROC", help='Metric to plot, choose between AUPRC, AUROC, F-score, Precision, Recall, for Precision and Recall you need to provide alpha')
args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
metric_string = args.metric
if args.sampling == 1:
    sampling_on = True
else:
    sampling_on = False

# check if metric_string is in the list of metrics
metrics = ["AUPRC", "AUROC", "F-score", "Precision", "Recall"]
if metric_string not in metrics:
    print("Please choose a metric from the following list: ", metrics)
    exit()
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

    for alpha in alphas:
        cutoff = np.percentile(background, (1 - alpha) * 100)
        tp = np.sum(np.array(pos_dist) > cutoff)
        fp = np.sum(np.array(neg_dist) > cutoff)
        fn = np.sum(np.array(pos_dist) <= cutoff)
        tn = np.sum(np.array(neg_dist) <= cutoff)

        prec.append(precision(tp, fp))
        rec.append(recall(tp, fn))
        fscores.append(f_score(tp, fp, fn))
        #mccs.append(mcc(tp, fp, tn, fn))
        cutoffs.append(cutoff)
    return prec, rec, fscores, mccs
def get_distributions(alphas, pos_dist, neg_dist, background):
    for alpha in alphas:
        cutoff = np.percentile(background, (1 - alpha) * 100)
        tp = np.sum(np.array(pos_dist) > cutoff)
        fp = np.sum(np.array(neg_dist) > cutoff)
        fn = np.sum(np.array(pos_dist) <= cutoff)
        tn = np.sum(np.array(neg_dist) <= cutoff)
    return tp, fp, fn, tn
all_methods = ["dMaSIF", "PInet", "GLINTER"]
all_pps = ["AA"]
all_datasets = ["CONTACT","PISA", "EPPIC"]

dict_groups = methods

data = dict()        
plt.figure(figsize=(6,6))
'''
for l, pp in enumerate(pps): 
    # Loop over each combination of method and preprocessing
    for method in methods:        
        experiment = f"{method}_{pp}"
        print(experiment)
        # Loop over each experiment
        for dataset in datasets:
            print(dataset)
            # Loop over each fold
            prec_rec = list()
            precs = list()
            recs = list()
            aucs = list()
            prs = list()
            f1s = list()
            alphas = [0.05]
            for fold in range(1, 6):
                if not os.path.exists(f"../results/{experiment}/{dataset}_back_fold{fold}.npy"):
                    #print(f"../results/{experiment}/{dataset}_back_fold{fold}.npy")
                    continue                   
                # Load data or calculate data (replace this part with your actual data loading)
                pos = np.load(f"../results/{experiment}/{dataset}_pos_fold{fold}.npy")
                neg = np.load(f"../results/{experiment}/{dataset}_neg_fold{fold}.npy")
                background = np.load(f"../results/{experiment}/{dataset}_back_fold{fold}.npy")

                # keep the same random samples in neg and background as pos length
                if sampling_on:
                    np.random.seed(42)
                    neg = np.random.choice(neg, len(pos), replace=False)
                    background = np.random.choice(background, len(pos), replace=False)
 
                if metric_string == "AUROC":
                    true = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
                    pred = np.concatenate((pos, neg))
                    fpr, tpr, thresholds = roc_curve(true, pred)
                    auc_score = roc_auc_score(true, pred)
                    aucs.append(auc_score)
                    

                if metric_string == "AUPRC":
                    prec, reca, threshold = precision_recall_curve(np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int))), np.concatenate((pos, neg)))
                    pr_score = auc(reca, prec)
                    prec_rec.append(pr_score)
                    
                #get_results(alphas, pos_dist, neg_dist)
                if metric_string == "Precision":
                    prec, rec, f1, mc = get_results(alphas, pos, neg, background)
                    precs.append(prec)
                if metric_string == "Recall":
                    prec, rec, f1, mc = get_results(alphas, pos, neg, background)
                    recs.append(rec)
                if metric_string == "F-score":
                    prec, rec, f1, mc = get_results(alphas, pos, neg, background)
                    f1s.append(f1)

            if metric_string == "AUPRC":
                if len(prec_rec) == 0:
                    prec_rec = [[0] for _ in range(5)]
                metric = prec_rec
                y_label = "AUPRC"

            if metric_string == "AUROC":
                if len(aucs) == 0:
                    aucs = [[0] for _ in range(5)]
                metric = aucs
                y_label = "AUROC"

                #mccs.append(mc)
            # for settings without results yet
            if metric_string == "Precision":
                if len(precs) == 0:
                    precs = [[0] for _ in range(5)]
                metric = precs
                y_label = "Precision"

            if metric_string == "Recall":
                if len(recs) == 0:
                    recs = [[0] for _ in range(5)]
                metric = recs
                y_label = "Recall"

            if metric_string == "F-score":
                if len(f1s) == 0:
                    f1s = [[0] for _ in range(5)]
                metric = f1s
                y_label = "F-score" 

            label_plot = f"{method} {dataset} - {pp}"

            
            #ax.errorbar(alphas, np.mean(metric, axis=0), yerr=np.std(tprs, axis=0), fmt='-', label=label_plot)
            data[label_plot] = metric


# Calculate the median and standard deviation for each key in data and store them in dictionaries

medians = []
stds = []
for key, values in data.items():
    data[key] = np.asarray(values)
    data[key] = [np.mean(values, axis=0), np.std(values, axis=0)]
'''
# save dict to file to faster load it
#np.save(f"results/plots/{metric_string}_data_{sampling_on}_AA.npy", data)
data = np.load(f"../results/plots/not_needed/{metric_string}_data_{sampling_on}_AA.npy", allow_pickle=True).item()

# Group data by tool (dMaSIF and PInet)
dMaSIF_data = {key: value for key, value in data.items() if 'dMaSIF' in key}
PInet_data = {key: value for key, value in data.items() if 'PInet' in key}
glinter_data = {key: value for key, value in data.items() if 'GLINTER' in key}

# Extract settings
settings = list(dMaSIF_data.keys())
num_settings = len(settings)
font_size = 16

fig, ax = plt.subplots()
bar_width = 0.1
color_map = dict()
all_colors = ['lightblue', 'lightskyblue', 'royalblue', 'palegreen', 'limegreen',\
               'seagreen', 'lightcoral', 'firebrick', 'maroon']

# get index from all_pps and all_datasets
for dataset in datasets:
    for pp in pps:
        color_map[f'{dataset} - {pp}'] = all_colors[all_datasets.index(dataset)*3+all_pps.index(pp)]


# Group data by dataset
contact_data = {key: value for key, value in dMaSIF_data.items() if 'CONTACT' in key}
pisa_data = {key: value for key, value in dMaSIF_data.items() if 'PISA' in key}
eppic_data = {key: value for key, value in dMaSIF_data.items() if 'EPPIC' in key}

# Combine the grouped data
grouped_data = {**contact_data, **pisa_data, **eppic_data}

# Iterate over the grouped data and plot the bars
for i, (key, value) in enumerate(grouped_data.items()):
    tool, setting = key.split(' ')[0], key.split(' ')[1:]
    setting_label = ' '.join(setting)
    ax.bar(i * bar_width, value[0], bar_width, yerr=value[1], color=color_map[setting_label], label=f'{setting_label}')

# Group data by dataset
contact_data = {key: value for key, value in PInet_data.items() if 'CONTACT' in key}
pisa_data = {key: value for key, value in PInet_data.items() if 'PISA' in key}
eppic_data = {key: value for key, value in PInet_data.items() if 'EPPIC' in key}

# Combine the grouped data
grouped_data = {**contact_data, **pisa_data, **eppic_data}
for i, (key, value) in enumerate(grouped_data.items()):
    tool, setting = key.split(' ')[0], key.split(' ')[1:]
    setting_label = ' '.join(setting)
    ax.bar((num_settings + i) * bar_width + bar_width, value[0], bar_width, yerr=value[1], color=color_map[setting_label], label=f'{setting_label}')

# Group data by dataset
contact_data = {key: value for key, value in glinter_data.items() if 'CONTACT' in key}
pisa_data = {key: value for key, value in glinter_data.items() if 'PISA' in key}
eppic_data = {key: value for key, value in glinter_data.items() if 'EPPIC' in key}

# Combine the grouped data
grouped_data = {**contact_data, **pisa_data, **eppic_data}
for i, (key, value) in enumerate(grouped_data.items()):
    tool, setting = key.split(' ')[0], key.split(' ')[1:]
    setting_label = ' '.join(setting)
    ax.bar((2*num_settings + i) * bar_width + 2*bar_width, value[0], bar_width, yerr=value[1], color=color_map[setting_label], label=f'{setting_label}')


# set xticks = dMaSIF settings + PInet settings
if len(methods) == 3:
    ax.set_xticks([num_settings*bar_width / 2 - 0.05, bar_width+1.5*num_settings*bar_width-0.05,2*bar_width+2.5*num_settings*bar_width-0.05])#, fontsize=font_size)
elif len(methods) == 2:
    ax.set_xticks([num_settings*bar_width / 2, bar_width+1.5*num_settings*bar_width])#, fontsize=font_size)
ax.set_xticklabels(methods, fontsize=font_size)


# Set labels and title
ax.set_ylabel("AU-ROC", fontsize=font_size, labelpad=15)
ax.set_xlabel('PPIIP method', fontsize = font_size, labelpad=15)
#ax.set_title('Values with Error Bars for Different Settings')

# Add legend
legend_labels = ["$D_{Con}$", "$D_{Engy}$", "$D_{Evol}$"]
ax.legend(legend_labels, loc='lower center', ncol=3, fontsize=font_size)
#show horizontal grid
#ax.yaxis.grid(True)

#set ylim
# AU-PRC ylim
#ax.set_ylim([0, 0.04])
#y_ticks = [0, 0.01, 0.02, 0.03, 0.04]
# AU-ROC ylim
ax.set_ylim([0, 1])
y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
plt.yticks(y_ticks, fontsize=font_size)
#ax.set_yticklabels(y_ticks, fontsize=font_size)
# Show the plot
if sampling_on:
    plt.savefig(f"../results/plots/bar/bar_plot_{metric_string}_samp.png", dpi=600, bbox_inches='tight')
else:
    plt.savefig(f"../results/plots/bar/bar_plot_{metric_string}_AAonly_0213.png", dpi=600, bbox_inches='tight')

# print the hight of each bar with its label
for i, (key, value) in enumerate(data.items()):
    print(key, value[0])
