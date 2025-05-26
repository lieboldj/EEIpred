import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import os
from itertools import combinations

metrics = ["MCC", "Fscore", "Precision", "Recall"]
methods = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER","RRI ProteinMAE",
           "PPMax dMaSIF", "PPMax PInet", "PPMax GLINTER","PPMax ProteinMAE",
           "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]
methods = ["PPMax dMaSIF", "PPMax PInet", "PPMax GLINTER","PPMax ProteinMAE",
           "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]
method_names = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.05]
all_q_matrix_directed = {}
all_p_tests = []

for alpha in alphas:
    if not os.path.exists(f"../results/plots/all_clust{alpha}.npy"):
        print("Missing file for alpha", alpha)
        continue

    data = np.load(f"../results/plots/all_clust{alpha}.npy", allow_pickle=True).item()

    # Prepare empty matrix holders
    if alpha == 0.05 and (metrics[0], alpha) in all_q_matrix_directed.keys():
        metrics = ["AUROC", "AUPRC"]
    else: 
        metrics = ["MCC", "Fscore", "Precision", "Recall"]
    for metric in metrics:
        all_q_matrix_directed[(metric, alpha)] = pd.DataFrame(np.nan, index=methods, columns=methods)

    for metric in metrics:   
        data_wil = dict()

        for method in method_names:
            for dataset in ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]:
                for pp in ["Max","DL"]:#, "Max", "DL"]:
                    key = f"{method} {dataset} - {pp} - {metric}"
                    if key in data:
                        values = data[key]
                        if pp == "AA":
                            new_key = f"RRI {method} - {metric}"
                        elif pp == "Max":
                            new_key = f"PPMax {method} - {metric}"
                        elif pp == "DL":
                            new_key = f"PPDL {method} - {metric}"
                        else:
                            continue
                        if new_key not in data_wil:
                            data_wil[new_key] = values
                        else:
                            data_wil[new_key] = np.concatenate((data_wil[new_key], values))
        
        for i, j in combinations(range(len(methods)), 2):
            k1 = f"{methods[i]} - {metric}"
            k2 = f"{methods[j]} - {metric}"
            if methods[i].split(" ")[0] == methods[j].split(" ")[0]:
                 continue
            elif methods[i].split(" ")[1] != methods[j].split(" ")[1]:
                continue

            if k1 not in data_wil or k2 not in data_wil:
                print(f"Missing data for {k1} or {k2}")
                continue

            values1 = data_wil[k1]
            values2 = data_wil[k2]

            # skip if nan
            mask = ~np.isnan(values1) & ~np.isnan(values2)
            values1 = values1[mask]
            values2 = values2[mask]
            if len(values1) < 15 or len(values2) < 15:
                print(f"Not enough data for {k1} or {k2}")
                continue

            stat, p = wilcoxon(values1, values2)
            mean_diff = np.mean(values1) - np.mean(values2)
            if mean_diff > 0:
                print(f"Method {methods[i]} is better than {methods[j]} for {metric} at alpha {alpha}")

            all_p_tests.append({
                "alpha": alpha,
                "metric": metric,
                "method1": methods[i],
                "method2": methods[j],
                "p": p,
                "mean_diff": mean_diff
            })

    # Global correction (across all metrics and method pairs)
p_vals = [r["p"] for r in all_p_tests]
_, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

# Store back into matrix
for row, q in zip(all_p_tests, q_vals):
    m1 = row["method1"]
    m2 = row["method2"]
    metric = row["metric"]
    alpha = row["alpha"]
    diff = row["mean_diff"]

    q_matrix = all_q_matrix_directed[(metric, alpha)]
    if diff > 0:
        q_matrix.loc[m1, m2] = q
    elif diff < 0:
        q_matrix.loc[m2, m1] = q

print("All p-values and q-values calculated and stored in matrices.")
# length of all_p_tests
print("Number of p-values:", len(all_p_tests))

for entry, q in zip(all_p_tests, q_vals):
    entry["q_value"] = q
output_file = "Max_DL_corrected_compact.csv"
method_names = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
metric_blocks = ["MCC", "Fscore", "Precision", "Recall", "AUROC", "AUPRC"]

q_values = np.zeros((22, len(method_names)))
mean_diffs = np.zeros((22, len(method_names)))
mean_d = 0

for i, method in enumerate(method_names):
    for j, metric in enumerate(metric_blocks):
        for k, alpha in enumerate(alphas):
            q_value = None
            mean_d = None
            for entry in all_p_tests:
                if entry["method1"] == f"PPDL {method}" and entry["method2"] == f"RRI {method}" and entry["metric"] == metric and entry["alpha"] == alpha:
                    q_value = entry.get("q_value", None)
                    mean_d = entry["mean_diff"]
                    break
                if entry["method1"] == f"PPMax {method}" and entry["method2"] == f"PPDL {method}" and entry["metric"] == metric and entry["alpha"] == alpha:
                    q_value = entry.get("q_value", None)
                    mean_d = entry["mean_diff"]
                    break
            if j < 4 : 
                q_values[(j*5)+k, i] = q_value if q_value is not None else np.nan
                mean_diffs[(j*5)+k, i] = entry["mean_diff"] if entry["mean_diff"] is not None else np.nan
            else:
                q_values[16+j, i] = q_value if q_value is not None else np.nan
                mean_diffs[16+j, i] = entry["mean_diff"] if entry["mean_diff"] is not None else np.nan
            

# plot heatmap_array
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

color_vals = np.copy(q_values)
color_vals[mean_diffs < 0] *= -1  # Negative if Max is better
# if mean diffs is nan make it ti np.nan
color_vals[np.isnan(q_values)] = np.nan

# Create custom colormap
cmap = ListedColormap(["white", "orange","navy", "white"])
bounds = [-1, -0.05, 0, 0.05, 1]
norm = BoundaryNorm(bounds, cmap.N)

#y_labels = 5*["MCC"]+5*["Fscore"]+5*["Precision"]+5*["Recall"]+["AUROC", "AUPRC"]
y_labels = 4*["1%", "2%", "3%", "4%", "5%"]+["AUROC     ", "AUPRC     "]

plt.figure(figsize=(10,9))

# set fontsize
plt.rcParams.update({'font.size': 15})

ax = sns.heatmap(
    color_vals,
    cmap=cmap,
    norm=norm,
    annot=np.round(q_values, 5),
    fmt=".5f",
    cbar=False,
    xticklabels=method_names,
    yticklabels=y_labels,
    linewidths=0.5,
    linecolor='gray',
)

# add text to the heatmap next to yticklabels, combine 5 yticklabels into one
plt.yticks(rotation=0)
for i in range(len(metric_blocks[:-2])):
    if metric_blocks[i] == "Fscore":
        plt.text(-0.3, i*5+2.5, "F-score", ha='right', va='center', fontsize=14, rotation=90)
    else:
        plt.text(-0.3, i*5+2.5, metric_blocks[i], ha='right', va='center', fontsize=14, rotation=90)

# Draw horizontal lines between metric blocks
for i in range(1, 5):
    plt.hlines(y = i * 5, xmin=-20, xmax=4, color='black', linewidth=3)

# add FDR above the ylabels
plt.text(0, -0.25, f"FDR", ha='right', va='center', fontsize=14)

plt.xlabel("PPIIP methods", labelpad=15)
plt.ylabel("Performance measures", labelpad=5)
plt.tight_layout()
plt.savefig("Max_DL_q-values.png", dpi=600)
