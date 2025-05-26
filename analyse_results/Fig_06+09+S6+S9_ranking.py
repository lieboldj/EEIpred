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
pick_dataset = "CLUST_EPPIC"
get_index_dataset = datasets.index(pick_dataset)

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
# Step 1: Get the max values across the last three dimensions (axis 2, 3, 4).
counts = np.zeros((ranked_results.shape[0], ranked_results.shape[1]), dtype=int)
for t in range(len(thresholds)):
    # Get the indices of the maximum values along the first dimension.
    max_indices = np.argmax(ranked_results[t], axis=0)  # Shape will be (3, 4, 5)

    # Count how many times each approach is the best.
    #counts = np.zeros(ranked_results[t].shape[0], dtype=int)  # Shape (5,)
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

# Add a legend
#ax.legend(ncol=2, loc='lower center', fontsize=font_size, bbox_to_anchor=(0.55, 0.1))

plt.tight_layout()
#Save figure
plt.savefig(f'{pps[0]}_ranked2.png', dpi=600)

exit()
# create bar plots with error bar for Fscore only but the 5 different thresholds
# take the mean over the last dimension and get the standard deviation over the last dimension
# Step 2: Calculate the mean and standard deviation of the F-scores.
mean_fscore = np.nanmean(ranked_results, axis=4)
std_fscore = np.nanstd(ranked_results, axis=4)

# Step 3: Plot the results.
fig, ax = plt.subplots()

# Select the relevant data
x_ticks = np.arange(mean_fscore.shape[0])  # dim0
mean_values_0 = mean_fscore[:, :, get_index_dataset, 2]    # second last dim: 1 = PISA, 0 = CONTACT 2 = EPPIC, last dim: 2 = Fscore, 1 = Recall, 0 = Precision, 3 = MCC
std_values_0 = std_fscore[:, :, get_index_dataset, 2]      # same here
# 5 thresholds, 4 methods, 3 datasets, 4 metrics, 5 test sets
#evals = ["Precision", "Recall", "Fscore", "MCC"]
# Bar Width to fit 5 bars in one group
bar_width = 0.15

# Plotting the bars with height mean and error bars with std
for i in range(mean_values_0.shape[1]):
    ax.bar(
        x_ticks + i * bar_width, 
        mean_values_0[:, i], 
        bar_width, 
        yerr=std_values_0[:, i], 
        label=methods[i], 
        color=colors[i]
    )
# Set the xticks and labels
ax.set_xticks(indices)
ax.set_xticklabels([f'{i+1}%' for i in indices], fontsize=font_size)
#ax.set_title(pps[0])

# Add labels and title
ax.set_xlabel('False Discovery Rate (FDR)', fontsize=font_size)
ax.set_ylabel('F-score', fontsize=font_size)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)

# Add a legend
ax.legend(ncol=2, loc='lower center', fontsize=font_size, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
# Save the figure
plt.savefig(f'{pps[0]}_{pick_dataset}_fscore.png', dpi=300)
