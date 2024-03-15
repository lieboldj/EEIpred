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

        thresholds = -1

        results_pos = []
        results_neg = []

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
                results_pos.append(np.sum(exon > thresholds) / size_exon)
                inter_exons.append(1)        

            else:
                exon = np.load(exon_pair_full)
                if exon.shape[0] > 100 or exon.shape[1] > 100:
                    continue
                size_exon = exon.size
                results_neg.append(np.sum(exon > thresholds) / size_exon)
 
        # Plot the histogram
        plt.figure(figsize=(8, 6))
        hist1, bins1, _ = plt.hist(results_pos, bins=20, color="b", alpha=0.7, label="Interacting exon pairs (test set)")
        bin_width1 = bins1[1] - bins1[0]
        bin_centers1 = bins1[:-1] + bin_width1 / 2
        plt.xticks(bin_centers1, ['{:.2f}-{:.2f}'.format(bins1[i], bins1[i+1]) for i in range(len(bins1)-1)], rotation=90)

        plt.xlabel('Number of dMaSIF scores > threshold')
        plt.ylabel('Total Number of exon pairs')
        plt.title('Distribution of dMaSIF scores over threshold {}'.format(thresholds))
        #plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        #plt.legend()

        plt.savefig(f"partions{m}_{thresholds}_{dataset}.png", dpi=600, bbox_inches="tight")

        plt.figure(figsize=(8, 6))
        hist2, bins2, _ = plt.hist(results_neg, bins=20, color="orange", alpha=0.7, label="Non-interacting exon pairs (training set)")
        bin_width2 = bins2[1] - bins2[0]
        bin_centers2 = bins2[:-1] + bin_width2 / 2
        plt.xticks(bin_centers2, ['{:.2f}-{:.2f}'.format(bins2[i], bins2[i+1]) for i in range(len(bins2)-1)], rotation=90)

        plt.xlabel('Number of dMaSIF scores > threshold')
        plt.ylabel('Total Number of exon pairs')
        plt.title('Distribution of dMaSIF scores over threshold {}'.format(thresholds))
        #plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.savefig(f"partions{m}_{thresholds}_{dataset}_neg.png", dpi=600, bbox_inches="tight")

        print("inter_exons: ", len(inter_exons))
        print("non_inter_exons: ", len(non_inter_exons))
