import numpy as np
import glob
import csv
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
# get histogram of exon pair shape
if os.path.exists("Tab_S1_shapes.pkl"):
    with open("Tab_S1_shapes.pkl", "rb") as f:
        shapes = pickle.load(f)
else:
    shapes = {}
    for dataset in datasets:
        print(dataset)
        shapes[dataset] = []
        for fold in range(1,6):
            pos = 0
            test_folder = f"../{method}/results/{dataset}/fold{fold}/test/"
            test_data = glob.glob(test_folder + "*.npy")

            for exon_pair in tqdm(test_data):
                exon_shape = np.load(exon_pair, allow_pickle=True).shape
                shapes[dataset].append(exon_shape)

    # save shapes to later plot
    with open("Tab_S1_shapes.pkl", "wb") as f:
        pickle.dump(shapes, f)

bin_step = 3
max_bin = 196
#bins = list(range(0, max_bin+1, bin_step)) + [max_bin, np.inf]  # last bin captures >200
bins = list(range(0, 100+1, bin_step))+ list(range(100+1, max_bin, bin_step)) + [max_bin, np.inf]
# Generate bin labels
# get histogram of shapes
labels = [f"[{bins[i]},{bins[i+1]-1}]" for i in range(0,len(bins)-2)] + [f">{max_bin}"]

for dataset in datasets:
    plt.figure()
    print(dataset)
    # get only the max value per tuple
    shapes[dataset] = [max(x) for x in shapes[dataset]]
    shapes[dataset] = np.array(shapes[dataset])
    
    
    # Bin the data manually to control histogram counts
    hist, _ = np.histogram(shapes[dataset], bins=bins)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(hist)), hist, tick_label=labels)
    plt.xticks(rotation=90)

    #plt.title(dataset)
    plt.xlabel("Number of residues in larger exon per exon pair")
    plt.ylabel("Absolute number of exon pairs")
    plt.tight_layout()
    plt.savefig(f"plots/hists/Tab_S1_{dataset}.png", dpi=600)
    plt.close()