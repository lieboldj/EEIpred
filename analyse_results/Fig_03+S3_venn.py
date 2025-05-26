import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib_venn import venn3
import matplotlib
import os
from Bio.PDB import PDBParser

def get_residues_name(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    unique_residues = list()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()[1]
                unique_residues.append(residue_id)
                #unique_residues.append((residue_id, residue.get_resname()))
    return unique_residues

venn = False
venn_prot_pairs = False
venn_new = False # only to create lists with interacting and non interacting exon pairs
venn_txt = False
venn_clust = True

save_non_int_exon = True

# datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
datasets = ["CONTACT", "PISA", "EPPIC"]
datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
exon = True
ppair = False
if venn_clust:
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
                    #elif ppair:
                    #    p1 = file.split("_")[0]
                    #    p2 = file.split("_")[1]
                    #    files.add(tuple(sorted((p1, p2))))
                    #else:
                    #    print("no exon or ppair")
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
        print(len(files), "files in total")
        print("total files: ", total_files)   
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
    print("Intersection of interacting exon pairs: ", len(intersection_int_exons))
    #print("Intersection of non-interacting exon pairs: ", len(intersection_non_int_exons))
    # create a venn diagramm for the exon pairs in the datasets
    plt.figure()
    print(datasets[2], datasets[1], datasets[0])
    print(len(dict_int_exons[datasets[2]]), len(dict_int_exons[datasets[1]]), len(dict_int_exons[datasets[0]]))
    print(len(dict_non_int_exons[datasets[2]]), len(dict_non_int_exons[datasets[1]]), len(dict_non_int_exons[datasets[0]]))
    
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
        plt.savefig('venn_ppairs_Fig3a.png', dpi=600)
    if exon: 
        plt.savefig('venn_exons_test2.png', dpi=600)
# create a venn diagramm for the protein pairs in the datasets
#from matplotlib_venn import venn2


################################################################################

# extra analysis not in paper
################################################################################
if venn:
    dict_int_exons = {}
    dict_non_int_exons = {}

    for dataset in ["CONTACT", "EPPIC", "PISA"]:
        print(dataset)
        path_to_data = "../data_collection/cv_splits/"
        path_to_pdb = "../PInet/data/exon/pdb/"
        path_to_map = "../data_collection/uniprot_EnsemblExonPDB_map/"

        filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
        df_pos = pd.read_csv(filename_pos, sep='\t')

        filename_neg = path_to_data + "{}/{}_negatives.txt".format(dataset, dataset)
        df_neg = pd.read_csv(filename_neg, sep='\t')

        non_interacting_pairs = set()
        interacting_pairs = set()

        #for index, row in tqdm(df_pos.iterrows()):
        #    exon_pair = tuple(sorted((row['EXON1'], row['EXON2'])))
        #    interacting_pairs.add(exon_pair)
        for index, row in tqdm(df_neg.iterrows()):
            exon_pair = tuple(sorted((row['EXON1'], row['EXON2'])))
            interacting_pairs.add(exon_pair)

        dict_int_exons[dataset] = interacting_pairs
        #dict_non_int_exons[dataset] = non_interacting_pairs

    # given all the interacting exon pairs from the different datasets, calculate the intersection
    intersection_int_exons = dict_int_exons["CONTACT"].intersection(dict_int_exons["EPPIC"]).intersection(dict_int_exons["PISA"])

    # create a venn diagramm for the exon pairs in the datasets
    plt.figure()
    out = venn3([dict_int_exons["EPPIC"], dict_int_exons["PISA"], dict_int_exons["CONTACT"]], ("$D_{Evol}$", "$D_{Engy}$", "$D_{Con}$"))

    for text in out.set_labels:
        text.set_fontsize(16)
            # set bold
        text.set_fontweight('bold')
    for text in out.subset_labels:
        text.set_fontsize(10)
    #plt.title('Interacting Exon Pairs')

    label_texts = [text for text in plt.gca().texts]

    for text in label_texts:
        if text.get_text() == "1621":  # Replace "5" with the actual text you want to move
            print("here")
            # Adjust the label's position
            text.set_x(text.get_position()[0] - 0.05)  # Move right
            #text.set_y(text.get_position()[1] - 0.1)  # Move up


    plt.tight_layout()
    plt.savefig('venn_nonint_exons_largefont.png', dpi=600)
    plt.savefig('venn_nonint_exons.pdf')

if venn_prot_pairs:
    dict_pps = {}

    for dataset in datasets:
        print(dataset)
        path_to_data = "../data_collection/cv_splits/"
        proteins_file = "/combined_info.txt"
        pdbs_file = "/combined.txt"

        prot_pairs = set()

        # read the proteins file
        with open(path_to_data + dataset + proteins_file, "r") as f:
            proteins = f.readlines()
            # split proteins at \t
            proteins = [p.split("\t") for p in proteins]
            # remove the newline character
            proteins = [(p[0], p[1].strip()) for p in proteins]
        
        # read the pdbs file
        #with open(path_to_data + dataset + pdbs_file, "r") as f:
        #    pdbs = f.readlines()
        #    # split pdbs at the two _ characters
        #    pdbs = [p.split("_") for p in pdbs]
        #    # remove the newline character
        #    pdbs = [(p[0], p[1], p[2].strip()) for p in pdbs]
#
        # iterate over proteins
        for i, (prot1, prot2) in enumerate(proteins):
            #pdb = pdbs[i]
            #combine the proteins and the pdb
            prot_pair = (prot1, prot2)#, pdb[0], pdb[1], pdb[2])
            prot_pair = sorted(prot_pair)
            # add protein pair to the set if not already in the set
            if tuple(prot_pair) not in prot_pairs:
                prot_pairs.add(tuple(prot_pair))
        # are there any duplicates?
        print("Number of duplicates: ", len(prot_pairs))
        print("Number of unique protein pairs: ", len(set(prot_pairs)))
        dict_pps[dataset] = prot_pairs

    # given all the interacting protein pairs from the different datasets, calculate the intersection
    intersection_prot_pairs = dict_pps[datasets[0]].intersection(dict_pps[datasets[2]]).intersection(dict_pps[datasets[1]])
#datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
    # create a venn diagramm for the protein pairs in the datasets
    plt.figure()
    out = venn3([dict_pps[datasets[2]], dict_pps[datasets[1]], dict_pps[datasets[0]]], ("$D_{Evol}$", "$D_{Engy}$", "$D_{Con}$"))
    for text in out.set_labels:
        text.set_fontsize(16)
            # set bold
        text.set_fontweight('bold')
    for text in out.subset_labels:
        text.set_fontsize(10)
    plt.tight_layout()
    for d in datasets:
        if "CLUST" in d:
            filen = d

    plt.savefig(f'venn_prot_pairs_{filen}.png', dpi=600)
    #plt.savefig('venn_prot_pairs.pdf')


if venn_new:
    dict_int_exons = {}
    dict_non_int_exons = {}
    for dataset in ["CONTACT", "EPPIC", "PISA"]:
        print(dataset)

        path_to_data = "../data_collection/cv_splits/"
        path_to_pdb = "../PInet/data/exon/pdb/"
        path_to_map = "../data_collection/uniprot_EnsemblExonPDB_map/"

        filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
        df_pos = pd.read_csv(filename_pos, sep='\t')

        non_interacting_pairs = set()
        interacting_pairs = set()

        prot_pairs_path = path_to_data + dataset +"/combined.txt"
        prot_pair_info_path = path_to_data + dataset +"/combined_info.txt"

        prot_pairs = list()
        prot_pairs_info = list()

        with open(prot_pairs_path, 'r') as f:
            for line in f:
                line = line.strip()
                prot_pairs.append(line)

        with open(prot_pair_info_path, 'r') as f:
            for line in f:
                line = line.strip()
                prot_pairs_info.append(line)

        for i in tqdm(range(len(prot_pairs))):
            p_pair = prot_pairs[i]
            p_pair_info = prot_pairs_info[i]

            prot, c1, c2 = p_pair.split("_")
            p1 = prot + "_" + c1
            p2 = prot + "_" + c2

            p1_info, p2_info = p_pair_info.split("\t")

            pdb_file1 = path_to_pdb + p1 + ".pdb"
            pdb_file2 = path_to_pdb + p2 + ".pdb"

            if not os.path.exists(pdb_file1) or not os.path.exists(pdb_file2):
                print("PDB file not found")

            map_file1 = path_to_map + "{}_{}.txt".format(p1_info, p1)
            map_file2 = path_to_map + "{}_{}.txt".format(p2_info,p2)

            if not os.path.exists(map_file1) or not os.path.exists(map_file2):
                print("Map file not found")

            # load the pdb files and get the residues
            residues1 = get_residues_name(pdb_file1)
            residues2 = get_residues_name(pdb_file2)

            # load the map files and get the exon residues
            exon_map_1 = np.genfromtxt(map_file1, delimiter="\t", dtype=str, skip_header=1)
            exon_map_2 = np.genfromtxt(map_file2, delimiter="\t", dtype=str, skip_header=1)

            exon_dict1 = {int(line[-1]): line[0] for line in exon_map_1 if line[-1] != "-"}
            exons1 = {v: [k for k, val in exon_dict1.items() if val == v] for v in set(exon_dict1.values())}

            exon_dict2 = {int(line[-1]): line[0] for line in exon_map_2 if line[-1] != "-"}
            exons2 = {v: [k for k, val in exon_dict2.items() if val == v] for v in set(exon_dict2.values())}

            exon1_index = {k: [residues1.index(v) for v in exons1[k]] for k in exons1.keys()}
            exon2_index = {k: [residues2.index(v) for v in exons2[k]] for k in exons2.keys()}

            for exon1 in exon1_index.keys():
                for exon2 in exon2_index.keys():
                    exon_pair = tuple(sorted((exon1, exon2)))
                    if (exon1, exon2) in zip(df_pos['EXON1'], df_pos['EXON2']) or \
                        (exon2, exon1) in zip(df_pos['EXON1'], df_pos['EXON2']):
                        interacting_pairs.add(exon_pair)
                    else:
                        
                        non_interacting_pairs.add(exon_pair)


        dict_int_exons[dataset] = interacting_pairs
        dict_non_int_exons[dataset] = non_interacting_pairs

        # save interacting and non-interacting exon pairs to file
        with open(f"{dataset}_int_exons.txt", "w") as f:
            for pair in interacting_pairs:
                f.write(f"{pair[0]}\t{pair[1]}\n")

        with open(f"{dataset}_non_int_exons.txt", "w") as f:
            for pair in non_interacting_pairs:
                f.write(f"{pair[0]}\t{pair[1]}\n")

if venn_txt:

    dict_int_exons = {}
    dict_non_int_exons = {}

    for dataset in ["CONTACT", "EPPIC", "PISA"]:
        print(dataset)
        with open(f"{dataset}_int_exons.txt", "r") as f:
            interacting_pairs = f.readlines()
            interacting_pairs = [tuple(line.strip().split("\t")) for line in interacting_pairs]

        with open(f"{dataset}_non_int_exons.txt", "r") as f:
            non_interacting_pairs = f.readlines()
            non_interacting_pairs = [tuple(line.strip().split("\t")) for line in non_interacting_pairs]

        tmp_set = set()
        for pair in interacting_pairs:
            tmp_set.add(pair)
        dict_int_exons[dataset] = tmp_set

    # given all the interacting exon pairs from the different datasets, calculate the intersection
    intersection_int_exons = dict_int_exons["CONTACT"].intersection(dict_int_exons["EPPIC"]).intersection(dict_int_exons["PISA"])

    # create a venn diagramm for the exon pairs in the datasets
    plt.figure()
    out = venn3([dict_int_exons["EPPIC"], dict_int_exons["PISA"], dict_int_exons["CONTACT"]], ("$D_{Evol}$", "$D_{Engy}$", "$D_{Con}$"))

    for text in out.set_labels:
        text.set_fontsize(16)
            # set bold
        text.set_fontweight('bold')
    for text in out.subset_labels:
        text.set_fontsize(10)
    #plt.title('Interacting Exon Pairs')

    label_texts = [text for text in plt.gca().texts]

    for text in label_texts:
        if text.get_text() == "1621":  # Replace "5" with the actual text you want to move
            print("here")
            # Adjust the label's position
            text.set_x(text.get_position()[0] - 0.05)  # Move right
            #text.set_y(text.get_position()[1] - 0.1)  # Move up


    plt.tight_layout()
    plt.savefig('venn_int_exons_poster.png', dpi=1000)
    plt.savefig('venn_nonint_exons.pdf')