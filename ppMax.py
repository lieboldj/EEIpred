import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataset = sys.argv[1] #"CONTACT, EPPIC, PISA"
    mode = sys.argv[2] #"train, test"
    method = sys.argv[3] #"dmasif, PInet, glinter"
    folds = sys.argv[4] #[1,2,3,4,5]" for all folds

    path_to_data = "data_collection/cv_splits/"

    filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
    df_pos = pd.read_csv(filename_pos, sep='\t')
    #split folds at ,
    folds = folds.split(",")
    for i in folds:
        print("Fold: ",i)
        inter_exons = []
        non_inter_exons = []
        all_exons = glob.glob("{}/results/{}/fold{}/{}/*".format(method, dataset,i,mode))
        all_exons = sorted(all_exons)

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
            if (df_pos[(df_pos["EXON1"] == exon1) & (df_pos["EXON2"] == exon2) & \
                    (df_pos["CHAIN1"] == (pdb+"_"+chain1)) & (df_pos["CHAIN2"] == pdb+"_"+chain2)].index.values).size > 0:
                #check for shape of the array > 500
                exon = np.load(exon_pair_full)
                if exon.shape[0] > 100 or exon.shape[1] > 100:
                    continue
                inter_exons.append(np.max(exon))
            elif (df_pos[(df_pos["EXON1"] == exon2) & (df_pos["EXON2"] == exon1) & \
                    (df_pos["CHAIN1"] == pdb+"_"+chain2) & (df_pos["CHAIN2"] == pdb+"_"+chain1)].index.values).size > 0:
                exon = np.load(exon_pair_full)
                if exon.shape[0] > 100 or exon.shape[1] > 100:
                    continue
                inter_exons.append(np.max(exon))
            else:
                exon = np.load(exon_pair_full)
                if exon.shape[0] > 100 or exon.shape[1] > 100:
                    continue
                non_inter_exons.append(np.max(exon))
        
        if method == "dmasif":
            if not os.path.exists("results/dMaSIF_Max/"):
                os.makedirs("results/dMaSIF_Max/")
            if mode == "test":
                np.save("results/dMaSIF_Max/{}_pos_fold{}.npy".format(dataset,i), inter_exons)
                np.save("results/dMaSIF_Max/{}_neg_fold{}.npy".format(dataset,i), non_inter_exons)
            else:
                np.save("results/dMaSIF_Max/{}_backPOS_fold{}.npy".format(dataset, i), inter_exons)
                np.save("results/dMaSIF_Max/{}_back_fold{}.npy".format(dataset, i), non_inter_exons)
        else:
            if not os.path.exists("results/{}_Max/".format(method)):
                os.makedirs("results/{}_Max/".format(method))

            if mode == "test":
                np.save("results/{}_Max/{}_pos_fold{}.npy".format(method,dataset,i), inter_exons)
                np.save("results/{}_Max/{}_neg_fold{}.npy".format(method,dataset,i), non_inter_exons)
            else:
                np.save("results/{}_Max/{}_backPOS_fold{}.npy".format(method, dataset, i), inter_exons)
                np.save("results/{}_Max/{}_back_fold{}.npy".format(method,dataset, i), non_inter_exons)

        print("inter_exons: ", len(inter_exons))
        print("non_inter_exons: ", len(non_inter_exons))
