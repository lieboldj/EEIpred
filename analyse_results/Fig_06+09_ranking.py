import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# set -p to AA for RRI and to DL for EEI using PPDL
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-m', '--method', type=str, default="dMaSIF,PInet,GLINTER,ProteinMAE", help='Methods to test')
parser.add_argument('-p', '--pp', type=str, default="DL", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CLUST_CONTACT,CLUST_PISA,CLUST_EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=bool, default=True, help='Sampling on or off')
parser.add_argument('-a', '--auroc', type=bool, default=False, help='Set to True if you want to know AUROC and AUPRC')
args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
sampling_on = args.sampling
auroc_on = args.auroc
evals = ["Precision", "Recall", "Fscore", "MCC"]

font_size = 16
thresholds = [i / 100 for i in range(1, 6)]
#ranked_results = np.zeros((len(thresholds), 9, 3))
roc_aucs = np.zeros((len(thresholds), 9))
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 10), sharex=True)
fig, ax1 = plt.subplots(1)#subplots(1, figsize=(12, 4), sharex=True)
thresholds = [0.01,0.02,0.03,0.04,0.05]
print(thresholds)
# for each threshold load the stored dict into one big dict
# 5 methods, 3 metrics, 5 thresholds
ranked_results = np.zeros((len(thresholds), len(methods), len(datasets), len(evals), 5))
for t, threshold in enumerate(thresholds):
        # load the results
        if datasets[0] == "CLUST_CONTACT":
            if os.path.exists(f"../results/plots/all_clust{threshold}.npy"):
                data = np.load(f"../results/plots/all_clust{threshold}.npy", allow_pickle=True).item() 

        elif datasets[0] == "CONTACT":
            if os.path.exists(f"../results/plots/all_{threshold}.npy"):
                data = np.load(f"../results/plots/all_{threshold}.npy", allow_pickle=True).item()
        for j, method in enumerate(methods):
            for i, dataset in enumerate(datasets):
                for k, metric in enumerate(evals):
                    if data[f"{method} {dataset} - {pps[0]} - {metric}"].flatten().shape[0] != 5:
                        print(metric)
                        print(dataset)
                        print("ERROR")
                        print(data[f"{method} {dataset} - {pps[0]} - {metric}"].flatten().shape)
                        ranked_results[t, j, i, k, :4] = data[f"{method} {dataset} - {pps[0]} - {metric}"].flatten()
                        ranked_results[t, j, i, k, -1] = np.nan
                        continue
                    ranked_results[t, j, i, k] = data[f"{method} {dataset} - {pps[0]} - {metric}"].flatten()

print(ranked_results.shape)
counts = np.zeros((ranked_results.shape[0], ranked_results.shape[1]), dtype=int)
for t in range(len(thresholds)):
    # Get the indices of the maximum values along the first dimension.
    max_indices = np.argmax(ranked_results[t], axis=0)  # Shape will be (3, 4, 5)

    for i in range(max_indices.shape[0]):
        for j in range(max_indices.shape[1]):
            for k in range(max_indices.shape[2]):
                counts[t, max_indices[i, j, k]] += 1

# Display the result
print("Counts of how often each approach is the best:")
print(counts)
# Define the number of groups and bars
num_groups, num_bars = counts.shape

# Define the x locations for the groups
indices = np.arange(num_groups)

# Define the colors for the bars
colors = ["#0000a7", "#008176", "#eecc16", "#d62728"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
# Create the plot
fig, ax = plt.subplots()

# Plot the bars
bottom = np.zeros(num_groups)
for i in range(num_bars):
    bars = ax.bar(indices, counts[:, i], bottom=bottom, color=colors[i], label=methods[i])#color=colors[i], 
    # Add text annotations
    for bar, value in zip(bars, counts[:, i]):
        if value > 2:
            height = bar.get_height()
            if value == 24:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + height / 2 + 2, 
                    f'{value}', 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=font_size
                )
            elif value == 22:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + height / 2 + 4, 
                    f'{value}', 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=font_size
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 
                    bar.get_y() + height / 2-0.25, 
                    f'{value}', 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=font_size
                )
    bottom += counts[:, i]

# Set the xticks and labels
ax.set_xticks(indices)
ax.set_xticklabels([f'{i+1}%' for i in indices], fontsize=font_size)
#ax.set_title(pps[0])
# Set the y-axis limit
ax.set_ylim(0, 60)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)

# Add labels and title
ax.set_xlabel('False Discovery Rate (FDR)', fontsize=font_size)
ax.set_ylabel('Number of performance tests', fontsize=font_size)

plt.tight_layout()
#Save figure
if args.pp == "AA":
    plt.savefig(f'Fig06_ranked.png', dpi=600)
elif args.pp == "DL":
    plt.savefig(f'Fig09_ranked.png', dpi=600)
