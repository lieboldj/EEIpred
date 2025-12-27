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
    #method_path = "../ProteinMAE/search/"

    if method == "dMaSIF":
        method_path = "dmasif"
    elif method == "GLINTER":
        method_path = "glinter"
    elif method == "ProteinMAE":
        method_path = "ProteinMAE/search"
    else:
        method_path = method
    #method_path = "dmasif"
    path_to_data = "data_collection/cv_splits/"
    pre_trained = ""
    if "pretrained" in dataset:
        pre_trained = "pretrained_"
        dataset = dataset.replace("pretrained_", "")

    filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset.split('_')[1], dataset.split('_')[1])
    df_pos = pd.read_csv(filename_pos, sep='\t')
    #split folds at ,
    folds = folds.split(",")
    for i in folds:
        print("Fold: ",i)
        inter_exons = []
        non_inter_exons = []
        print("Path: ", "{}/results/{}{}/fold{}/{}/*".format(method_path, pre_trained, dataset,i,mode))
        all_exons = glob.glob("{}/results/{}{}/fold{}/{}/*".format(method_path, pre_trained, dataset,i,mode))
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
            method = "dMaSIF"
        elif method == "glinter":
            method = "GLINTER"

        if not os.path.exists("results/{}_Max/".format(method)):
            os.makedirs("results/{}_Max/".format(method))

        if mode == "test":
            np.save("results/{}_Max/pdbV6_{}_test_pos_fold{}.npy".format(method,dataset,i), inter_exons)
            np.save("results/{}_Max/pdbV6_{}_test_neg_fold{}.npy".format(method,dataset,i), non_inter_exons)
        elif mode == "val":
            np.save("results/{}_Max/{}_val_pos_fold{}.npy".format(method, dataset, i), inter_exons)
            np.save("results/{}_Max/{}_val_neg_fold{}.npy".format(method, dataset, i), non_inter_exons)
        else:
            np.save("results/{}_Max/{}_train_pos_fold{}.npy".format(method, dataset, i), inter_exons)
            np.save("results/{}_Max/{}_train_neg_fold{}.npy".format(method,dataset, i), non_inter_exons)
        
        if method == "dMaSIF":
            method = "dmasif"
        elif method == "GLINTER":
            method = "glinter"
        print("inter_exons: ", len(inter_exons))
        print("non_inter_exons: ", len(non_inter_exons))
