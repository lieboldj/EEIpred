

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
parser.add_argument('-p', '--pp', type=str, default="DLs", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CONTACT,PISA,EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AUROC')
parser.add_argument('-mr','--metric', type=str, default="AUPRC", help='Metric to plot, choose between AUPRC, AUROC, F-score, Precision, Recall, for Precision and Recall you need to provide alpha')
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

all_methods = ["dMaSIF", "PInet", "GLINTER"]
all_pps = ["DLs","AA", "Max"]
all_datasets = ["CONTACT","PISA", "EPPIC"]

dict_groups = methods

data = dict()        
plt.figure(figsize=(6,6))

# save dict to file to faster load it
#np.save(f"../results/plots/{metric_string}_data_{sampling_on}_AA.npy", data)
data = np.load(f"../results/plots/not_needed/{metric_string}_data_{sampling_on}_DLs.npy", allow_pickle=True).item()


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
    print(value[0], value[1])
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
if metric_string == "AUROC":
    ax.set_ylabel("AU-ROC", fontsize=font_size, labelpad=15)
    ax.set_ylim([0, 1])
    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
elif metric_string == "AUPRC":
    ax.set_ylabel("AU-PRC", fontsize=font_size, labelpad=15)
    ax.set_ylim([0, 0.6])
    y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#ax.set_ylabel("AU-ROC", fontsize=font_size, labelpad=15)
ax.set_xlabel('PPIIP method', fontsize = font_size, labelpad=15)
#ax.set_title('Values with Error Bars for Different Settings')

# Add legend
legend_labels = ["$D_{Con}$", "$D_{Engy}$", "$D_{Evol}$"]
ax.legend(legend_labels, loc='lower center', ncol=3, fontsize=font_size)
#show horizontal grid
#ax.yaxis.grid(True)
# AU-ROC ylim

plt.yticks(fontsize=font_size)
#AU-PRC ylim
#ax.set_ylim([0, 0.6])
#y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ax.set_yticklabels(y_ticks, fontsize=font_size)


# Show the plot
if sampling_on:
    plt.savefig(f"../results/plots/bar/bar_plot_{metric_string}_samp.png", dpi=600, bbox_inches='tight')
else:
    plt.savefig(f"../results/plots/bar/bar_plot_{metric_string}_DLs0213.png", dpi=600, bbox_inches='tight')

# print the hight of each bar with its label
for i, (key, value) in enumerate(data.items()):
    print(key, value[0])
