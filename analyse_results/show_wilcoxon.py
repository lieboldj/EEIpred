#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel, shapiro
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from statsmodels.stats.multitest import multipletests
import sys

metric = sys.argv[1] if len(sys.argv) > 1 else "Recall"

# Provided data
if metric == "Recall":
    df = pd.read_csv("../results/pre-calc/sig_test_recall.csv")
elif metric == "Precision":
    df = pd.read_csv("../results/pre-calc/sig_test_precision.csv")
elif metric == "F-score":
    df = pd.read_csv("../results/pre-calc/sig_test_f-score.csv")
else:
    print("Metric not supported")
    exit()
for alpha in [0.01, 0.02, 0.03, 0.04, 0.05]:
    #alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    metric = "AU-PRC"
    df = pd.read_csv(f"../results/pre-calc/{metric}_{alpha}.csv")

    sig_test = df.to_numpy()[:, 1:]
    sig_test = sig_test
    # get :3 and 6: for EEI
    #sig_test = np.concatenate((sig_test[:3], sig_test[6:]))
    methods = ["dMaSIF", "PInet", "GLINTER"]
    methods = ["PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER"]
    methods_EEI = ["PPMax dMaSIF", "PPMax PInet", "PPMax GLINTER",\
                    "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER"]
    methods_EEI = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER",\
                    "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER"]
    methodsALL = ["RRI dMaSIF", "RRI PInet", "RRI GLINTER",\
                    "PPMax dMaSIF", "PPMax PInet", "PPMax GLINTER",\
                    "PPDL dMaSIF", "PPDL PInet", "PPDL GLINTER"]

    len_comp = sig_test.shape[0]
    conf_matrix = np.ones((len_comp, len_comp))
    conf_p = np.ones((len_comp, len_comp))
    print(sig_test)
    for i in range(len_comp):
        for j in range(len_comp):
            if i == j:
                conf_matrix[i, j] = 1
                continue
            values1 = sig_test[i]
            values2 = sig_test[j]

            # Perform Wilcoxon test
            statistic, p_value = wilcoxon(values1, values2)

            # Determine which method has a higher median
            median_diff = np.mean(values1) - np.mean(values2)
            if median_diff > 0:
                conf_matrix[i,j] = p_value
                conf_p[i,j] = p_value

            else:
                conf_matrix[i,j] = 1
                conf_p[i,j] = 1
    # Apply Benjamini-Hochberg procedure
    reject, q_value, _, _ = multipletests(conf_p.flatten(), method='fdr_bh')
    #print(conf_matrix)
    conf_matrix = q_value.reshape((len_comp, len_comp))

    if len_comp == 6:
        methods = methods_EEI
    elif len_comp == 9:
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
    plt.savefig(f"../results/plots/wilcoxon/{metric}_wil_{alpha}.png", bbox_extra_artists=(y_lab,), bbox_inches='tight', dpi=600)

    #write data to file
    data.to_csv(f"../results/plots/wilcoxon/{metric}_wil_{alpha}.csv")