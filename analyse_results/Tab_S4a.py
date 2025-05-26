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
methods = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER","RRI ProteinMAE",
           "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]
methods = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER","RRI ProteinMAE"]
method_names = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
alphas = [0.05, 0.05, 0.01, 0.02, 0.03, 0.04]
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
                for pp in ["AA"]:#, "Max", "DL"]:
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
        print("Number of combinations:", len(list(combinations(range(len(methods)), 2))))
        for i, j in combinations(range(len(methods)), 2):
            print("Hello")
            k1 = f"{methods[i]} - {metric}"
            k2 = f"{methods[j]} - {metric}"
            print(k1,k2)
            if methods[i].split(" ")[0] != methods[j].split(" ")[0]:
                 continue
            elif methods[i].split(" ")[1] == methods[j].split(" ")[1]:
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
            print("need to print more")
            print(mean_diff)

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

import csv
for entry, q in zip(all_p_tests, q_vals):
    entry["q_value"] = q
    # Ensure method order is get q-value only for better performing method
    if entry["mean_diff"] < 0:
        entry["method1"], entry["method2"] = entry["method2"], entry["method1"]
output_file = "RRI_corrected_compact.csv"
method_names = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
metric_blocks = ["MCC", "Fscore", "Precision", "Recall", "AUROC", "AUPRC"]


alphas = [0.05, 0.01, 0.02, 0.03, 0.04]
with open(output_file, "w", newline='') as f:

    for alpha in alphas:
        f.write(f"FDR={int(alpha*100)}%")
        for metric in metric_blocks:
            if metric in ["AUROC", "AUPRC"]:
                if alpha == 0.05:
                    f.write(f"\n{metric}\n")
                else:
                    continue
            else:
                f.write(f"\n{metric}\n")

            for method in method_names:
                m1 = f"RRI {method}"
                f.write(f"{m1},")
                for method1 in method_names:  
                    m2 = f"RRI {method1}"
                    if m1 == m2:
                        f.write(f",")
                        continue
                    q_value = None
                    for entry in all_p_tests:
                        if entry["method1"] == m1 and entry["method2"] == m2 and entry["metric"] == metric and entry["alpha"] == alpha:
                            q_value = entry.get("q_value", None)
                            break
                    if q_value is not None:
                        print(f"Found q-value for {m1} vs {m2}: {q_value}")
                        f.write(f"{q_value}")
                    f.write(",")
                f.write("\n")
        f.write("\n")
