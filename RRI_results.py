import sys
from Bio.PDB import PDBParser
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def get_unique_residues(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    unique_residues = list()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()[1]
                unique_residues.append(residue_id)
    return unique_residues


if __name__ == "__main__":
    dataset = sys.argv[1] #"CONTACT, EPPIC, PISA"
    path_to_pdb = "PInet/data/exon/pdb/"
    path_to_map = "data_collection/uniprot_EnsemblExonPDB_map/"
    path_to_data = "data_collection/cv_splits/"
    #rr_cutoff = 4
    pre_trained = ""
    if "pretrained" in dataset:
        pre_trained = "pretrained_"
        dataset = dataset.replace("pretrained_", "")

    filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
    filename_neg = path_to_data + "{}/{}_negatives.txt".format(dataset, dataset)

    df_pos = pd.read_csv(filename_pos, sep='\t')
    df_neg = pd.read_csv(filename_neg, sep='\t')

    mode = sys.argv[2] #"train, test"
    method = sys.argv[3] #"dMaSIF, PInet, glinter" ProteinMAE
    #if len(sys.argv) > 5:
    #    method_path = sys.argv[6] #"dmasif, PInet, glinter" ../ProteinMAE/search
    if method == "dMaSIF":
        method_path = "dmasif"
    elif method == "GLINTER":
        method_path = "glinter"
    elif method == "ProteinMAE":
        method_path = "ProteinMAE/search"
    else:
        method_path = method

    cutoff = int(sys.argv[5]) #4, 6, 8
    folds = sys.argv[4] #[1,2,3,4,5]" for all folds
    
    if cutoff == 6:
        df_aa = pd.read_csv("data_collection/aa_interactions6_24-01-04.txt", sep='\t')
    elif cutoff == 4:
        df_aa = pd.read_csv("data_collection/aa_interactions4.txt", sep='\t')
    elif cutoff == 8:
        df_aa = pd.read_csv("data_collection/aa_interactions8.txt", sep='\t')
    else:
        print(f"cutoff {cutoff} not supported, we take default cutoff = 6")
        df_aa = pd.read_csv("data_collection/aa_interactions6_24-01-04.txt", sep='\t')
        cutoff = 6
    #df_aa = pd.read_csv(f"data_collection/aa_interactions{rr_cutoff}.txt", sep='\t')
    #split folds at ,
    folds = folds.split(",")
    for i in folds:
        if method != "GLINTER":
            df_test = pd.read_csv(path_to_data + "{}/{}{}.txt".format(dataset, mode, i), sep='_', header=None)
            df_testinfo = pd.read_csv(path_to_data + "{}/{}_info{}.txt".format(dataset, mode, i), sep='\t', header=None)
        else:
            df_test = pd.read_csv(path_to_data + "{}/{}{}_glinter.txt".format(dataset, mode, i), sep='_', header=None)
            df_testinfo = pd.read_csv(path_to_data + "{}/{}_info{}_glinter.txt".format(dataset, mode, i), sep='\t', header=None)
        df_test.columns = ['PDB', 'Chain1', 'Chain2']
        df_testinfo.columns = ['UniProt1', 'UniProt2']
        exon_pair_counter = 0
        exon_pair_pos_counter = 0
        exon_pair_neg_counter = 0
        inter_aa = []
        not_inter_aa = []

        for j in tqdm(range(len(df_test))):
            pdb = df_test["PDB"][j]
            chain1 = df_test["Chain1"][j]
            chain2 = df_test["Chain2"][j]
            uniprot1 = df_testinfo["UniProt1"][j]
            uniprot2 = df_testinfo["UniProt2"][j]
            # check whether there is a map file for this PDB
            map_file1 = path_to_map + "{}_{}_{}.txt".format(uniprot1, pdb, chain1)
            map_file2 = path_to_map + "{}_{}_{}.txt".format(uniprot2, pdb, chain2)

            data1 = np.genfromtxt(map_file1, delimiter="\t", dtype=str, skip_header=1)
            # create dict which residue belongs to which exon from 1
            exon_dict1 = {int(line[-1]): line[0] for line in data1 if line[-1] != "-"}
            exons1 = {v: [k for k, val in exon_dict1.items() if val == v] for v in set(exon_dict1.values())}
            exon_dict1b = {int(line[-3]): line[0] for line in data1 if line[-1] != "-"}
            exons1b = {v: [k for k, val in exon_dict1b.items() if val == v] for v in set(exon_dict1b.values())}
            # load exon2 information
            data2 = np.genfromtxt(map_file2, delimiter="\t", dtype=str, skip_header=1)
            # create dict which residue belongs to which exon from 2
            exon_dict2 = {int(line[-1]): line[0] for line in data2 if line[-1] != "-"}
            exons2 = {v: [k for k, val in exon_dict2.items() if val == v] for v in set(exon_dict2.values())}
            exon_dict2b = {int(line[-3]): line[0] for line in data2 if line[-1] != "-"}
            exons2b = {v: [k for k, val in exon_dict2b.items() if val == v] for v in set(exon_dict2b.values())}
            # get unique residues from both protein pdb files individually with their residue number
            residues1_pdb = get_unique_residues(path_to_pdb + "{}_{}.pdb".format(pdb, chain1))
            residues2_pdb = get_unique_residues(path_to_pdb + "{}_{}.pdb".format(pdb, chain2))            

            # map exon dict values to index of residues1_pdb_hydro and residues2_pdb_hydro in new dict
            # keep the name of the exon in dict as key but map to index of residues1_pdb_hydro and residues2_pdb_hydro
            exon1_index = {k: [residues1_pdb.index(v) for v in exons1[k]] for k in exons1.keys()}
            exon2_index = {k: [residues2_pdb.index(v) for v in exons2[k]] for k in exons2.keys()}
            exon1_indexb = {k: [residues1_pdb.index(v) for v in exons1[k]] for k in exons1b.keys()}
            exon2_indexb = {k: [residues2_pdb.index(v) for v in exons2[k]] for k in exons2b.keys()}

            sample_pos_count = 0
            prot_pdb1 = uniprot1 + "_" + pdb + "_" + chain1
            prot_pdb2 = uniprot2 + "_" + pdb + "_" + chain2

            # save each exon pair in protein_pair_random in npy file
            for exon1 in exon1_index.keys():
                for exon2 in exon2_index.keys():
                    correct_order = True
                    if os.path.exists("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(method_path,pre_trained,dataset,i,mode, uniprot1, uniprot2, pdb, chain1, pdb, chain2, exon1, exon2)):
                        prediction = np.load("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(method_path,pre_trained,dataset,i,mode, uniprot1, uniprot2, pdb, chain1, pdb, chain2, exon1, exon2))
                    elif os.path.exists("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_big.npy".format(method_path,pre_trained,dataset,i,mode, uniprot1, uniprot2, pdb, chain1, pdb, chain2, exon1, exon2)):
                        prediction = np.load("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_big.npy".format(method_path,pre_trained,dataset,i,mode, uniprot1, uniprot2, pdb, chain1, pdb, chain2, exon1, exon2))
                    elif os.path.exists("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_big.npy".format(method_path,pre_trained,dataset,i,mode, uniprot2, uniprot1, pdb, chain2, pdb, chain1, exon2, exon1)):
                        prediction = np.load("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}_big.npy".format(method_path,pre_trained,dataset,i,mode, uniprot2, uniprot1, pdb, chain2, pdb, chain1, exon2, exon1))
                        correct_order = False
                    elif os.path.exists("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(method_path,pre_trained,dataset,i,mode, uniprot2, uniprot1, pdb, chain2, pdb, chain1, exon2, exon1)):
                        prediction = np.load("{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(method_path,pre_trained,dataset,i,mode, uniprot2, uniprot1, pdb, chain2, pdb, chain1, exon2, exon1))
                        correct_order = False
                    else:
                        print("no prediction", "{}/results/{}{}/fold{}/{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(method_path,pre_trained,dataset,i, mode, uniprot1, uniprot2, pdb, chain1, pdb, chain2, exon1, exon2))
                        exit()
                        continue

                    true_index1 = df_aa[(df_aa["exon1"] == exon1) & (df_aa["exon2"] == exon2) & \
                                         (df_aa["chain1"] == prot_pdb1) & (df_aa["chain2"] == prot_pdb2)]["exon1_PDBResNumCIF"].values
                    true_index2 = df_aa[(df_aa["exon1"] == exon1) & (df_aa["exon2"] == exon2) & \
                                            (df_aa["chain1"] == prot_pdb1) & (df_aa["chain2"] == prot_pdb2)]["exon2_PDBResNumCIF"].values
                    if not(len(true_index1)>0 and len(true_index2)>0):
                        true_index1a = df_aa[(df_aa["exon1"] == exon2) & (df_aa["exon2"] == exon1) & \
                                             (df_aa["chain1"] == prot_pdb2) & (df_aa["chain2"] == prot_pdb1)]["exon1_PDBResNumCIF"].values
                        true_index2a = df_aa[(df_aa["exon1"] == exon2) & (df_aa["exon2"] == exon1) & \
                                             (df_aa["chain1"] == prot_pdb2) & (df_aa["chain2"] == prot_pdb1)]["exon2_PDBResNumCIF"].values
                    
                    if len(true_index1)>0 and len(true_index2)>0:
                        try:
                            true_indexa = [exons1b[exon1].index(i) for i in true_index1]
                            true_indexb = [exons2b[exon2].index(i) for i in true_index2]
                            resi_inter = prediction[(true_indexa, true_indexb)]
                            mask = np.ones_like(prediction, dtype=bool)
                            mask[true_indexa, true_indexb] = False
                            resi_not_inter = prediction[mask]
                            inter_aa.extend(resi_inter.flatten())
                            not_inter_aa.extend(resi_not_inter.flatten())
                        except:
                            print(correct_order)
                            print("Error1", prot_pdb1, prot_pdb2, exon1, exon2, true_index1, true_index2)
                        continue
                    elif len(true_index1a)>0 and len(true_index2a)>0:
                        try:
                            true_indexa = [exons1b[exon1].index(i) for i in true_index2a]
                            true_indexb = [exons2b[exon2].index(i) for i in true_index1a]
                            resi_inter = prediction[(true_indexa, true_indexb)]
                            mask = np.ones_like(prediction, dtype=bool)
                            mask[true_indexa, true_indexb] = False
                            resi_not_inter = prediction[mask]
                            inter_aa.extend(resi_inter.flatten())
                            not_inter_aa.extend(resi_not_inter.flatten())
                        except:
                            print("Error2", prot_pdb1, prot_pdb2, exon1, exon2, true_index1a, true_index2a)
                    else:
                        not_inter_aa.extend(prediction.flatten())
                    # append resi_inter and resi_not_inter to list
                    
        #save inter_aa and not_inter_aa to file
        # make dir if not exists
        if method == "dmasif":
            method = "dMaSIF"
        if cutoff == 6:
            if not os.path.exists(f"results/{method}_AA/"):
                os.makedirs(f"results/{method}_AA/")
            if mode == "test":
                np.save("results/{}_AA/{}{}_test_pos_fold{}.npy".format(method,pre_trained,dataset, i), inter_aa)
                np.save("results/{}_AA/{}{}_test_neg_fold{}.npy".format(method,pre_trained,dataset, i), not_inter_aa)
            if mode == "train":
                np.save("results/{}_AA/{}{}_train_neg_fold{}.npy".format(method,pre_trained,dataset, i), not_inter_aa)
                np.save("results/{}_AA/{}{}_train_pos_fold{}.npy".format(method,pre_trained,dataset, i), inter_aa)
                        # get the indices of the residues in the exon
            if mode == "val":
                np.save("results/{}_AA/{}{}_val_neg_fold{}.npy".format(method,pre_trained,dataset, i), not_inter_aa)
                np.save("results/{}_AA/{}{}_val_pos_fold{}.npy".format(method,pre_trained,dataset, i), inter_aa)
        else:
            if not os.path.exists(f"results/{method}_AA_{cutoff}/"):
                os.makedirs(f"results/{method}_AA_{cutoff}/")
            if mode == "test":
                np.save("results/{}_AA_{}/{}{}_test_pos_fold{}.npy".format(method,cutoff,pre_trained,dataset, i), inter_aa)
                np.save("results/{}_AA_{}/{}{}_test_neg_fold{}.npy".format(method,cutoff,pre_trained,dataset, i), not_inter_aa)
            if mode == "train":
                np.save("results/{}_AA_{}/{}{}_train_neg_fold{}.npy".format(method,cutoff,pre_trained,dataset, i), not_inter_aa)
                np.save("results/{}_AA_{}/{}{}_train_pos_fold{}.npy".format(method,cutoff,pre_trained,dataset, i), inter_aa)
                        # get the indices of the residues in the exon
            if mode == "val":
                np.save("results/{}_AA_{}/{}{}_val_neg_fold{}.npy".format(method,cutoff,pre_trained,dataset, i), not_inter_aa)
                np.save("results/{}_AA_{}/{}{}_val_pos_fold{}.npy".format(method,cutoff,pre_trained,dataset, i), inter_aa)
        if method == "dMaSIF":
            method = "dmasif"