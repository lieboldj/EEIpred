import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib_venn import venn3
import matplotlib
import os

save_non_int_exon = True
load_from_file = True

datasets = ["CONTACT", "PISA", "EPPIC"]
datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
exon = True
ppair = False

if load_from_file:
    if exon:
        if "CLUST" in datasets[0]:
            dict_int_exons = np.load("exon_int_exons.npy", allow_pickle=True).item()
            dict_non_int_exons = np.load("exon_non_int_exons.npy", allow_pickle=True).item()
        else:
            dict_int_exons = np.load("exon_train_int_exons.npy", allow_pickle=True).item()
            dict_non_int_exons = np.load("exon_train_non_int_exons.npy", allow_pickle=True).item()
    elif ppair:
        if "CLUST" in datasets[0]:
            dict_int_exons = np.load("ppair_int_exons.npy", allow_pickle=True).item()
            dict_non_int_exons = np.load("ppair_non_int_exons.npy", allow_pickle=True).item()
        else:
            dict_int_exons = np.load("ppair_train_int_exons.npy", allow_pickle=True).item()
            dict_non_int_exons = np.load("ppair_train_non_int_exons.npy", allow_pickle=True).item()
else:
    dict_int_exons = {}
    dict_non_int_exons = {}
    for dataset in datasets:
        print(dataset)
        path_to_data = "../data_collection/cv_splits/"
        path_to_pdb = "../PInet/data/exon/pdb/"
        path_to_exons = "/cosybio/project/EEIP/EEIP/dmasif/results/"
        filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
        df_pos = pd.read_csv(filename_pos, sep='\t')
        filename_neg = path_to_data + "{}/{}_negatives.txt".format(dataset, dataset)
        df_neg = pd.read_csv(filename_neg, sep='\t')
        non_interacting_pairs = set()
        interacting_pairs = set()
        # get all files names in the directory
        files = set()
        all_exons = set()
        total_files = 0
        if exon: 
            for fold in range(1, 6):
                file_list = os.listdir(path_to_exons + dataset + "/fold" + str(fold) + "/test/")
                print(len(file_list), "files in fold ", fold)
                total_files += len(file_list)
                for file in file_list:
                    #if exon: 
                    # if files does not exist in files, add it
                    exon1 = file.split("_")[6]
                    exon2 = file.split("_")[7]
                    exon2 = exon2.split(".")[0]
                    prot1 = file.split("_")[0]
                    prot2 = file.split("_")[1]
                    pdb1 = file.split("_")[2] + "_" + file.split("_")[3]
                    pdb2 = file.split("_")[4] + "_" + file.split("_")[5]
                    #sort exon1 and exon2
                    if exon1 > exon2:
                        exon1, exon2 = exon2, exon1
                        prot1, prot2 = prot2, prot1
                        pdb1, pdb2 = pdb2, pdb1
                    exon_complete = tuple((exon1, exon2, prot1, prot2, pdb1, pdb2))
                    all_exons.add(exon_complete)
                    exon_pair = tuple(((exon1, exon2, pdb1, pdb2)))
                    #exon_pair = tuple(sorted((exon_complete)))
                    #exon_pair = exon_complete
                    if exon_pair not in files:
                        files.add(exon_complete)
                    else:
                        print("exon pair already in files: ", exon_pair)
        elif ppair:
            # load the protein pairs from combined file
            with open(path_to_data + dataset + "/combined_train_info.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    # split the line at the two _ characters
                    p1, p2 = line.split("\t")
                    # remove the new line character from p2
                    p2 = p2.strip()
                    #p1 = p1 + "_" + p2
                    #p2 = p1 + "_" + p3
                    # add the protein pair to the set
                    files.add(tuple(sorted((p1, p2))))

        # save the files to a file
        extra_set = set()
        if exon: 
            counter_else = 0
            for index, row in tqdm(df_pos.iterrows()):
                if row['EXON1'] > row['EXON2']:
                    e1 = row['EXON2']
                    e2 = row['EXON1']
                    p1 = row['CHAIN2']
                    p2 = row['CHAIN1']
                    up1 = row['UNIPROT2']
                    up2 = row['UNIPROT1']
                else:
                    e1 = row['EXON1']
                    e2 = row['EXON2']
                    p1 = row['CHAIN1']
                    p2 = row['CHAIN2']
                    up1 = row['UNIPROT1']
                    up2 = row['UNIPROT2']
                #exon_pair = tuple(sorted((row['EXON1'], row['EXON2'])))
                exon_pair = tuple((e1, e2, up1, up2, p1, p2))
                
                if exon_pair in files:
                    extra_set.add(exon_pair)
                    # check for all_exons if any entry has the first two elements of the tuple
                    #for entry in all_exons:
                    #    if exon_pair[0] in entry and exon_pair[1] in entry:
                    #        interacting_pairs.add(entry)
                    interacting_pairs.add(exon_pair)
                
            #for index, row in tqdm(df_neg.iterrows()):
            #    exon_pair = tuple(sorted((row['EXON1'], row['EXON2'])))
            for exon_pair in tqdm(files):
                if exon_pair not in extra_set:
                    #for entry in all_exons:
                    #    if exon_pair[0] in entry and exon_pair[1] in entry:
                    #        non_interacting_pairs.add(exon_pair)
                    non_interacting_pairs.add(exon_pair)
        elif ppair:
            interacting_pairs = files
            non_interacting_pairs = files
        dict_int_exons[dataset] = interacting_pairs
        dict_non_int_exons[dataset] = non_interacting_pairs
# given all the interacting exon pairs from the different datasets, calculate the intersection
intersection_int_exons = dict_int_exons[datasets[0]].intersection(dict_int_exons[datasets[2]]).intersection(dict_int_exons[datasets[1]])
if save_non_int_exon:
    intersection_int_exons = dict_non_int_exons[datasets[0]].intersection(dict_non_int_exons[datasets[2]]).intersection(dict_non_int_exons[datasets[1]])
#print("Intersection of non-interacting exon pairs: ", len(intersection_non_int_exons))
# create a venn diagramm for the exon pairs in the datasets
plt.figure()

if save_non_int_exon:
    out = venn3([dict_non_int_exons[datasets[2]], dict_non_int_exons[datasets[1]], dict_non_int_exons[datasets[0]]], ("$D_{Evol}$", "$D_{Engy}$", "$D_{Con}$"))
else:
    out = venn3([dict_int_exons[datasets[2]], dict_int_exons[datasets[1]], dict_int_exons[datasets[0]]], ("$D_{Evol}$", "$D_{Engy}$", "$D_{Con}$"))
for text in out.set_labels:
    text.set_fontsize(16)
        # set bold
    text.set_fontweight('bold')
for text in out.subset_labels:
    text.set_fontsize(10)
#plt.title('Interacting Exon Pairs')
plt.tight_layout()
if ppair:
    if "CLUST" in datasets[0]:
        plt.savefig('Fig3_venn_ppairs.png', dpi=600)
    else:
        plt.savefig('FigS3_venn_ppairs.png', dpi=600)
if exon: 
    if "CLUST" in datasets[0]:
        if save_non_int_exon:
            plt.savefig('Fig3_venn_exons_non_int.png', dpi=600)
        else: 
            plt.savefig('Fig3_venn_exons_int.png', dpi=600)
    else:
        if save_non_int_exon:
            plt.savefig('FigS3_venn_exons_non_int.png', dpi=600)
        else:
            plt.savefig('FigS3_venn_exons_int.png', dpi=600)
