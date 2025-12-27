

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
# set -p to AA for RRI and to DL for EEI using PPDL
# set -mr to AUPRC or AUROC
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-mh', '--method', type=str, default="dMaSIF,PInet,GLINTER,ProteinMAE", help='Methods to test')#dMaSIF,PInet,GLINTER,ProteinMAE
parser.add_argument('-p', '--pp', type=str, default="AA", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CLUST_CONTACT,CLUST_PISA,CLUST_EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AUROC')
parser.add_argument('-mr','--metric', type=str, default="AUROC", help='Metric to plot, choose between AUPRC or AUROC')
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
    numerator = np.float128(TP * TN) - np.float128(FP * FN)
    # check if denominator is 0
    TPFP = np.float128(TP + FP)
    TPFN = np.float128(TP + FN)
    TNFP = np.float128(TN + FP)
    TNFN = np.float128(TN + FN)
    denominator = np.float128(np.sqrt(TPFP * TPFN * TNFP * TNFN))
    return numerator / denominator

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
        if tp + fp == 0 or tp + fn == 0 or tn + fp == 0 or tn + fn == 0:
            mccs.append(0)
        else:
            mccs.append(mcc(tp, fp, tn, fn))
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

def plot_methods(ax, methods, datasets, data_dicts, num_settings, bar_width, font_size, color_map, x_ticks_labels):
    for method_index, (method, data) in enumerate(data_dicts.items()):
        grouped_data = {key: value for key, value in data.items() if any(dataset in key for dataset in datasets)}
        for i, (key, value) in enumerate(grouped_data.items()):
            tool, setting = key.split(' ')[0], key.split(' ')[1:]
            setting_label = ' '.join(setting)
            print(setting_label, setting)
            bar_position = (method_index * (num_settings + 1) + i) * bar_width
            #ax.bar(bar_position, value[0], bar_width, yerr=value[1], color=color_map[setting_label], label=f'{setting_label}')
            # Plot the bar (mean only, no error bar)
            ax.bar(bar_position, value[0], bar_width, color=color_map[setting_label], label=f'{setting_label}')

            # Overlay individual data points (assuming `individual_scores` contains them)
            # Shift x slightly to avoid overlapping with the bar edges
            #(np.random.rand(len(value[1]))) * bar_width/5 + bar_position
            jitter = np.linspace(-bar_width / 3, bar_width / 3, len(value[1]))
            ax.scatter(jitter + bar_position,
                       value[1],
                       color=color_map[setting_label], alpha=1, s=15, 
                       label='_nolegend_', edgecolors='black',
                       linewidth=0.3, zorder=2, marker='o')


            if (args.pp == "DL" or args.pp == "Max") and args.metric == "AUPRC" and datasets[0] == "CONTACT":
                if i % 3 == 0: # CONTACT
                    hline_height = 0.2209280904
                elif i % 3 == 1: # PISA
                    hline_height = 0.3477447084
                elif i % 3 == 2: # EPPIC
                    hline_height = 0.3609405375
                ax.hlines(hline_height, bar_position - bar_width/2, bar_position + bar_width/2, color='gray', linestyle='solid', label='_nolegend_')

            elif args.pp == "AA" and args.metric == "AUPRC" and datasets[0] == "CONTACT":
                if i % 3 == 0: # CONTACT
                    hline_height = 0.002656900248
                elif i % 3 == 1: # PISA
                    hline_height = 0.002456525148
                elif i % 3 == 2: # EPPIC
                    hline_height = 0.002704997385

            elif args.pp == "AA" and args.metric == "AUPRC" and datasets[0] == "CLUST_CONTACT":
                if i % 3 == 0: # CLUST_CONTACT
                    hline_height = 0.00261823615266007
                elif i % 3 == 1: # CLUST_PISA
                    hline_height = 0.002893609026835760
                elif i % 3 == 2: # CLUST_EPPIC
                    hline_height = 0.00308511574578032

            elif (args.pp == "Max" or args.pp == "DL") and args.metric == "AUPRC" and datasets[0] == "CLUST_CONTACT":
                if i % 3 == 0: # CONTACT
                    hline_height = 0.20944126556714912
                elif i % 3 == 1: # PISA
                    hline_height = 0.39322301024428685
                elif i % 3 == 2: # EPPIC
                    hline_height = 0.3615733736762481
            #ax.axvline(bar_position + bar_width, color='black', linewidth=0.5, linestyle='--', ymax=hline_height)
            if args.metric == "AUPRC":
                ax.hlines(hline_height, bar_position - bar_width/2, bar_position + bar_width/2, color='gray', linestyle='solid', label='_nolegend_')
           
    
    # Set xticks and labels
    tick_positions = []
    for i, method in enumerate(methods):
        tick_position = (i * (num_settings + 1) + num_settings / 2) * bar_width - 0.5*bar_width
        tick_positions.append(tick_position)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_ticks_labels, fontsize=font_size)


    # set lim to 0-1
    if metric_string == "AUROC" and args.pp == "DL":
        ax.set_ylim([0, 0.8])
    elif metric_string == "AUPRC" and args.pp == "AA":
        ax.set_ylim([0, 0.05])
    else:
        # get the max value of the data
        max_value = 0
        for key, values in data_dicts.items():
            for key, value in values.items():
                max_value = max(max_value, np.max(value[0]+np.std(value[1])))
        ax.set_ylim([0, max_value + max_value/10])
    #plt.yticks(y_ticks, fontsize=font_size)

all_methods = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
all_pps = ["AA", "Max", "DL"]
all_datasets = ["CLUST_CONTACT","CLUST_PISA", "CLUST_EPPIC"]

dict_groups = methods

#plt.figure(figsize=(6,6))
data = dict()   
use_all = True 
# if there is CLUST in dataset names
if "CLUST" in datasets[0]:
    if os.path.exists(f"../results/plots/all_clust{alphas[0]}.npy") and use_all:
        data = np.load(f"../results/plots/all_clust{alphas[0]}.npy", allow_pickle=True).item()  
else:
    if os.path.exists(f"../results/plots/all_{alphas[0]}.npy") and use_all:
        data = np.load(f"../results/plots/all_{alphas[0]}.npy", allow_pickle=True).item()

# to generate intermediate results for faster plotting set to True
if False:
    for l, pp in enumerate(pps): 
        # Loop over each combination of method and preprocessing
        for method in methods: 
            fig, axes = plt.subplots(5, 3, figsize=(27, 9))      
            experiment = f"{method}_{pp}"
            print(experiment)
            # Loop over each experiment
            for d, dataset in enumerate(datasets):
                print(dataset)
                # Loop over each fold
                prec_rec = list()
                precs = list()
                recs = list()
                aucs = list()
                prs = list()
                f1s = list()
                mccs = list()
                #alphas = [0.01]
                for fold in tqdm(range(1, 6)):                
                    # Load data or calculate data (replace this part with your actual data loading)
                    pos = np.load(f"../results/{experiment}/{dataset}_test_pos_fold{fold}.npy")
                    neg = np.load(f"../results/{experiment}/{dataset}_test_neg_fold{fold}.npy")

                    background = np.load(f"../results/{experiment}/{dataset}_train_neg_fold{fold}.npy")

                    #pos_val = np.load(f"../results/{experiment}/{dataset}_val_pos_fold{fold}.npy")
                    #neg_val = np.load(f"../results/{experiment}/{dataset}_val_neg_fold{fold}.npy")
                    # concatenate background and val
                        #background = np.concatenate((background, val))
                    if conf_matrix:
                        # plot the confusion matrix
                        y_true = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
                        y_pred = np.concatenate((pos, neg))
                        cm = confusion_matrix(y_true, y_pred > np.percentile(background, (1 - alphas[0]) * 100))
                        # make cm relative
                        cm_rel = cm / cm.sum(axis=1)[:, np.newaxis]

                        annot_text = np.empty_like(cm).astype(str)
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                annot_text[i, j] = f"{cm[i, j]} ({cm_rel[i, j]:.2f})"

                        heatmap = sns.heatmap(cm_rel.astype(float), annot=annot_text, fmt='', cmap='Blues', ax = axes[fold-1, d],
                                    xticklabels=['Pred. Non-Int.', 'Pred. Int'], 
                                    yticklabels=['True Non-Int', 'True Int'], cbar=False)
                        #plt.title(f"Confusion matrix for {method} {dataset} - {pp}")
                        if dataset == all_datasets[0]:
                            dat_set = "$D_{Con}$"
                        elif dataset == all_datasets[1]:
                            dat_set = "$D_{Engy}$"
                        elif dataset == all_datasets[2]:
                            dat_set = "$D_{Evol}$"
                        axes[fold-1, d].set_title(f"{dat_set} (fold {fold})")

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
                        print(auc_score)
                        aucs.append(auc_score)


                    elif metric_string == "AUPRC":
                        prec, reca, threshold = precision_recall_curve(np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int))), np.concatenate((pos, neg)))
                        pr_score = auc(reca, prec)
                        prec_rec.append(pr_score)

                    #get_results(alphas, pos_dist, neg_dist)
                    else:
                        prec, reca, threshold = precision_recall_curve(np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int))), np.concatenate((pos, neg)))
                        pr_score = auc(reca, prec)
                        prec_rec.append(pr_score)
                        prec, rec, f1, mc = get_results(alphas, pos, neg, background)
                        precs.append(prec)
                        recs.append(rec)
                        f1s.append(f1)
                        mccs.append(mc)
                        true = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
                        pred = np.concatenate((pos, neg))
                        fpr, tpr, thresholds = roc_curve(true, pred)
                        auc_score = roc_auc_score(true, pred)
                        aucs.append(auc_score)

                label_plot = f"{method} {dataset} - {pp}"
                if metric_string == "AUROC":
                    if len(aucs) == 0:
                        aucs = [[0] for _ in range(5)]
                    y_label = "AUROC"
                    data[label_plot + " - AUROC"] = aucs


                if use_all:
                    if len(aucs) == 0:
                        aucs = [[0] for _ in range(5)]
                    y_label = "AUROC"
                    data[label_plot + " - AUROC"] = aucs

                    if len(prec_rec) == 0:
                        prec_rec = [[0] for _ in range(5)]
                    y_label = "AUPRC"
                    data[label_plot + " - AUPRC"] = prec_rec

                    if len(precs) == 0:
                        precs = [[0] for _ in range(5)]
                    y_label = "Precision"
                    data[label_plot + " - Precision"] = precs

                    if len(recs) == 0:
                        recs = [[0] for _ in range(5)]
                    y_label = "Recall"
                    data[label_plot + " - Recall"] = recs

                    if len(f1s) == 0:
                        f1s = [[0] for _ in range(5)]
                    y_label = "F-score" 
                    data[label_plot + " - Fscore"] = f1s

                    if len(mccs) == 0:
                        mccs = [[0] for _ in range(5)]
                    y_label = "MCC"
                    data[label_plot + " - MCC"] = mccs


                #ax.errorbar(alphas, np.mean(metric, axis=0), yerr=np.std(tprs, axis=0), fmt='-', label=label_plot)
                 #metric

                for key, values in data.items():
                    data[key] = np.asarray(values)
            if conf_matrix:
                cbar = fig.colorbar(heatmap.collections[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
                plt.savefig(f"plots/cm_{method}_{pp}_rel.png", dpi=600, bbox_inches='tight')
                plt.close()

    # Calculate the median and standard deviation for each key in data and store them in dictionaries
    #print(data)
    medians = []
    stds = []

        # if key has ProteinMAE, then do it

        #data[key] = [np.mean(values, axis=0), np.std(values, axis=0)]
    #print(data)

    # save dict to file to faster load it
    np.save(f"../results/plots/all_clust{alphas[0]}_4.npy", data)
    exit()

# Extract settings
# only keep the keys that have one "p" in it
pp = pps[0]
data = {key: value for key, value in data.items() if pp in key}

#print(data)
font_size = 18

fig, ax = plt.subplots()
bar_width = 0.1
color_map = dict()
all_colors = ['lightblue', 'lightskyblue', 'royalblue', 'palegreen', 'limegreen',\
               'seagreen', 'lightcoral', 'firebrick', 'maroon']

# get index from all_pps and all_datasets
for dataset in datasets:
    for pp in pps:
        color_map[f'{dataset} - {pp} - {metric_string}'] = all_colors[all_datasets.index(dataset)*3]#+all_pps.index(pp)]
        print(f'{dataset} - {pp} - {metric_string}')
# get mean and std for each key in data
for key,values in data.items():
    #data[key] = [np.mean(values, axis=0), np.std(values, axis=0)]
    data[key] = [np.mean(values, axis=0), values]

# only keep the keys that have the metric_string in it
data = {key: value for key, value in data.items() if metric_string in key}

# keep only the keys that have the pp in it

data = {key: value for key, value in data.items() if pp + " -" in key}
settings = list({key: value for key, value in data.items() if 'dMaSIF' in key}.keys())
num_settings = len(settings)

data_dicts = dict()
x_ticks_labels = []
# Group data by tool (dMaSIF and PInet)
if "dMaSIF" in methods:
    dMaSIF_data = {key: value for key, value in data.items() if 'dMaSIF' in key}
    # Group data by dataset
    dMaSIF_contact_data = {key: value for key, value in dMaSIF_data.items() if all_datasets[0] in key}
    dMaSIF_pisa_data = {key: value for key, value in dMaSIF_data.items() if all_datasets[1] in key}
    dMaSIF_eppic_data = {key: value for key, value in dMaSIF_data.items() if all_datasets[2] in key}

    dMaSIF_grouped_data = {**dMaSIF_contact_data, **dMaSIF_pisa_data, **dMaSIF_eppic_data}
    data_dicts["dMaSIF"] = dMaSIF_grouped_data

    # add x_ticks_labels
    x_ticks_labels.append("dMaSIF")

if "PInet" in methods:
    PInet_data = {key: value for key, value in data.items() if 'PInet' in key}
    # Group data by dataset
    PInet_contact_data = {key: value for key, value in PInet_data.items() if all_datasets[0] in key}
    PInet_pisa_data = {key: value for key, value in PInet_data.items() if all_datasets[1] in key}
    PInet_eppic_data = {key: value for key, value in PInet_data.items() if all_datasets[2] in key}

    PInet_grouped_data = {**PInet_contact_data, **PInet_pisa_data, **PInet_eppic_data}
    data_dicts["PInet"] = PInet_grouped_data

    # add x_ticks_labels
    x_ticks_labels.append("PInet")

if "GLINTER" in methods:
    glinter_data = {key: value for key, value in data.items() if 'GLINTER' in key}
    # Group data by dataset
    glinter_contact_data = {key: value for key, value in glinter_data.items() if all_datasets[0] in key}
    glinter_pisa_data = {key: value for key, value in glinter_data.items() if all_datasets[1] in key}
    glinter_eppic_data = {key: value for key, value in glinter_data.items() if all_datasets[2] in key}

    glinter_grouped_data = {**glinter_contact_data, **glinter_pisa_data, **glinter_eppic_data}
    data_dicts["GLINTER"] = glinter_grouped_data

    # add x_ticks_labels
    x_ticks_labels.append("GLINTER")

if "ProteinMAE" in methods and not "pretrained" in methods:
    proteinmae_data = {key: value for key, value in data.items() if 'ProteinMAE' in key and not "pretrained" in key}
    # Group data by dataset
    proteinmae_contact_data = {key: value for key, value in proteinmae_data.items() if all_datasets[0] in key}
    proteinmae_pisa_data = {key: value for key, value in proteinmae_data.items() if all_datasets[1] in key}
    proteinmae_eppic_data = {key: value for key, value in proteinmae_data.items() if all_datasets[2] in key}

    proteinmae_grouped_data = {**proteinmae_contact_data, **proteinmae_pisa_data, **proteinmae_eppic_data}
    data_dicts["ProtMAE"] = proteinmae_grouped_data

    # add x_ticks_labels
    x_ticks_labels.append("   ProteinMAE")

number_tools = len(methods)

plot_methods(ax, methods, datasets, data_dicts, num_settings, bar_width, font_size, color_map, x_ticks_labels)
# Set labels and title
if metric_string == "AUROC":
    ax.set_ylabel("AU-ROC", fontsize=font_size, labelpad=15)
elif metric_string == "AUPRC":
    ax.set_ylabel("AU-PRC", fontsize=font_size, labelpad=15)
else:
    ax.set_ylabel(metric_string, fontsize=font_size, labelpad=15)
ax.set_xlabel('PPIIP method', fontsize=font_size, labelpad=15)
# make sure the yticks are .1f

if args.metric == "AUROC":
    pass
    #ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8], fontsize=font_size)
else: 
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)
# Add legend
legend_labels = ["$D_{Con}$", "$D_{Engy}$", "$D_{Evol}$"]
#if args.metric == "AUPRC":
#    ax.legend(legend_labels, loc='lower center', ncol=3, fontsize=font_size, bbox_to_anchor=(0.5, -0.5))
#else:
#    ax.legend(legend_labels, loc='lower center', ncol=1, fontsize=font_size, bbox_to_anchor=(1.15, 0.67))
plt.savefig(f"Fig5a.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.savefig(f"Fig5a.pdf")

