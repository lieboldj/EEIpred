#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel, shapiro
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from statsmodels.stats.multitest import multipletests
import sys
import os

metric = sys.argv[1] if len(sys.argv) > 1 else "Recall"

# Provided data
if metric == "Recall":
    df = pd.read_csv("../results/pre-calc/sig_test_recall.csv")
elif metric == "Precision":
    df = pd.read_csv("../results/pre-calc/sig_test_precision.csv")
elif metric == "F-score":
    metric = "Fscore"
    df = pd.read_csv("../results/pre-calc/sig_test_f-score.csv")

else:
    print("Metric not supported")
    exit()
#for alpha in [0.01, 0.02, 0.03, 0.04, 0.05]:
for alpha in [0.05]:
    #alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    #df = pd.read_csv(f"../results/pre-calc/{metric}_{alpha}.csv")
    if os.path.exists(f"../results/plots/all_{alpha}.npy"):
        data = np.load(f"../results/plots/all_{alpha}.npy", allow_pickle=True).item()  
    else:
        print("problems exiting")
        exit()

    # get :3 and 6: for EEI
    #sig_test = np.concatenate((sig_test[:3], sig_test[6:]))
    methods = ["dMaSIF", "PInet", "GLINTER", "ProteinMAE"]
    #methods = ["PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]
    methods_EEI = ["PPMax dMaSIF", "PPMax PInet", "PPMax GLINTER", "PPMax ProteinMAE",\
                    "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]
    methods_EEI = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER","RRI ProteinMAE",\
                    "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]
    methodsALL = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER","RRI ProteinMAE",\
                    "PPMax dMaSIF", "PPMax PInet", "PPMax GLINTER","PPMax ProteinMAE",\
                    "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER", "PPDL ProteinMAE"]

    # get metric
    data = {key: value for key, value in data.items() if metric in key}
    data_wil = dict()
    # in data, merge the values of the same method and pp but different datasets
    for method in methods:
        for dataset in ["CONTACT", "PISA", "EPPIC"]:
            for pp in ["AA", "Max", "DL"]:
                key = f"{method} {dataset} - {pp} - {metric}"
                if key in data:
                    values = data[key]
                    if pp == "AA":
                        new_key = f"RRI {method} - {metric}"
                    if pp == "Max":
                        new_key = f"PPMax {method} - {metric}"
                    if pp == "DL":
                        new_key = f"PPDL {method} - {metric}"
                    if new_key not in data_wil:
                        data_wil[new_key] = values
                    else:
                        data_wil[new_key] = np.concatenate((data_wil[new_key], values))

    len_comp = len(methodsALL)
    conf_matrix = np.ones((len_comp, len_comp))
    conf_p = np.ones((len_comp, len_comp))
    for i, m1 in enumerate(methodsALL):
        for j, m2 in enumerate(methodsALL):
            if i == j:
                conf_matrix[i, j] = 1
                continue
            k1 = f"{m1} - {metric}"
            k2 = f"{m2} - {metric}"
            values1 = data_wil[k1]
            values2 = data_wil[k2]

            # Perform Wilcoxon test
            statistic, p_value = wilcoxon(values1, values2)

            # Determine which method has a higher median
            median_diff = np.mean(values1) - np.mean(values2)
            if median_diff > 0:
                conf_matrix[i,j] = p_value[0]
                conf_p[i,j] = p_value[0]

            else:
                conf_matrix[i,j] = 1
                conf_p[i,j] = 1
    # Apply Benjamini-Hochberg procedure
    reject, q_value, _, _ = multipletests(conf_p.flatten(), method='fdr_bh')
    #print(conf_matrix)
    conf_matrix = q_value.reshape((len_comp, len_comp))


    methods = methodsALL
    data = pd.DataFrame(
        conf_matrix,
        index=methods,
        columns=methods
    )

    #plt.figure(figsize = (3,3))
    plt.figure()

    #colors = sns.color_palette("Spectral", n_colors=8)

    colors = ['#FF0000', '#0000FF']
    cmap = ListedColormap(colors, N=2)
    intervals = [0, 0.05, 1]
    norm = BoundaryNorm(intervals, len(colors))
    sns.set(font_scale=1.5)
    heatmap = sns.heatmap(data, fmt=".5f", cbar_kws={'label': 'q-value', "aspect" : 2}, #"shrink": 0.15, "aspect": 1
                          cmap=cmap, norm=norm, linewidths=0.5, linecolor='lightgray',square=True)  # Set fontsize to 10 annot_kws={'size': 10},  

    colorbar = heatmap.collections[0].colorbar

    # move each tick to the middle of interval
    ticks_intervals = [intervals[i] + (intervals[i+1]-intervals[i])/2 for i in range(len(intervals)-1)]
    colorbar.set_ticks(ticks_intervals)
    #labels = ['[0,1x10$^{-7}$]', '(1x10$^{-7}$,1x10$^{-6}$]', '(1x10$^{-6}$,1x10$^{-5}$]', '(1x10$^{-5}$,0.0001]', '(0.0001,0.001]', '(0.001,0.01]', '(0.01,0.05]', '(0.05,1]']
    labels = ['[0,0.05]', '(0.05,1]']
    #labels = intervals
    colorbar.set_ticklabels(labels)


    plt.xlabel("PPIIP method")
    y_lab = plt.ylabel("PPIIP method")
    plt.tight_layout()
    #plt.title("Wilcoxon Signed-Rank Test p-values")
    if alpha == 0.05:
        plt.savefig(f"../results/plots/wilcoxon/{metric}_wil4_{alpha}.png", bbox_extra_artists=(y_lab,), bbox_inches='tight', dpi=600)

    #write data to file
        data.to_csv(f"../results/plots/wilcoxon/{metric}_wil4_{alpha}.csv")