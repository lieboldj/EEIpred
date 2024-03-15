import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = "CONTACT"#sys.argv[1] #"CONTACT, EPPIC, PISA"
    mode = "test"#sys.argv[2] #"train, test"
    method = "dmasif"#sys.argv[3] #"dmasif, PInet, glinter"
    folds = "1,2,3,4,5"#sys.argv[4] #[1,2,3,4,5]" for all folds

    path_to_data = "../data_collection/cv_splits/"

    filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
    df_pos = pd.read_csv(filename_pos, sep='\t')
    #split folds at ,
    folds = folds.split(",")
    for m in folds:
        print("Fold: ",m)
        inter_exons = []
        non_inter_exons = []
        all_exons = glob.glob("../{}/results/{}/fold{}/{}/*".format(method, dataset,m,mode))
        all_exons = sorted(all_exons)

        thresholds = np.linspace(-4,5,20)

        results_pos = np.zeros(len(thresholds)+1)
        results_neg = np.zeros(len(thresholds)+1)

        for exon_pair_full in tqdm(all_exons):
            exon_pair = exon_pair_full.split("/")[-1].split("_")
            uniprot1 = exon_pair[0]
            uniprot2 = exon_pair[1]
            pdb = exon_pair[2]
            chain1 = exon_pair[3]
            chain2 = exon_pair[5]
            exon1 = exon_pair[6]

            # make sure to evaluate all exons
            if len(exon_pair) == 8:
                exon2 = exon_pair[7][:-4]
            else:
                exon2 = exon_pair[7]            
            # check if the exon pair is in df_pos you only need to check for exon1 in "EXON1" and exon2 in "EXON2" in one line
            if ((df_pos[(df_pos["EXON1"] == exon1) & (df_pos["EXON2"] == exon2) & \
                    (df_pos["CHAIN1"] == (pdb+"_"+chain1)) & (df_pos["CHAIN2"] == pdb+"_"+chain2)].index.values).size > 0) or\
                       ((df_pos[(df_pos["EXON1"] == exon2) & (df_pos["EXON2"] == exon1) & \
                    (df_pos["CHAIN1"] == pdb+"_"+chain2) & (df_pos["CHAIN2"] == pdb+"_"+chain1)].index.values).size > 0) :
                #check for shape of the array > 500
                exon = np.load(exon_pair_full)
                if exon.shape[0] > 100 or exon.shape[1] > 100:
                    continue
                size_exon = exon.size
                old_threshold = -np.inf
                sum_partion = 0
                all_partions = []
                for j, threshold in enumerate(thresholds):
                    results_pos[j] += np.sum((old_threshold < exon) & (exon <= threshold)) / size_exon
                    old_threshold = threshold
                results_pos[-1] = np.sum((old_threshold < exon)) / size_exon   
                inter_exons.append(1)        

            else:
                exon = np.load(exon_pair_full)
                if exon.shape[0] > 100 or exon.shape[1] > 100:
                    continue
                size_exon = exon.size
                old_threshold = -np.inf
                sum_partion = 0
                all_partions = []
                for j, threshold in enumerate(thresholds):
                    results_neg[j] += np.sum((old_threshold < exon) & (exon <= threshold)) / size_exon
                    old_threshold = threshold
                results_neg[-1] = np.sum((old_threshold < exon)) / size_exon   

        # Create intervals for the x-axis ticks
        x_intervals = ['<'+str(thresholds[0])] + ['({:.2f},{:.2f}]'.format(thresholds[i], thresholds[i + 1]) for i in range(len(thresholds)-1)] + ['>'+str(thresholds[-1])]

        # Plot the bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(x_intervals, results_pos/len(inter_exons), color='skyblue', edgecolor='black')

        plt.xlabel('dMaSIF scores')
        plt.ylabel('Average percentage of scores in interval')
        plt.title('dMaSIF CONTACT Fold {} Test Set'.format(m))
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.savefig(f"partions{m}.png", dpi=600, bbox_inches="tight")

        plt.figure(figsize=(10, 6))
        plt.bar(x_intervals, results_neg, color='skyblue', edgecolor='black')

        plt.xlabel('dMaSIF scores')
        plt.ylabel('Average percentage of scores in interval')
        plt.title('dMaSIF CONTACT Fold {} Test Set'.format(m))
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.savefig(f"partions{m}_neg.png", dpi=600, bbox_inches="tight")

        print("inter_exons: ", len(inter_exons))
        print("non_inter_exons: ", len(non_inter_exons))
