import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

#get the residues of the exons with name 
def get_residues_name(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    unique_residues = list()
    residue_types = list()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()[1]
                unique_residues.append(residue_id)
                residue_types.append(residue.get_resname())
                #unique_residues.append((residue_id, residue.get_resname()))
    return unique_residues, residue_types

if __name__ == "__main__":
    #dataset = sys.argv[1] #"CONTACT, EPPIC, PISA"
    #characteristic = sys.argv[2] #hydrophobic, hydrophilic, amphipathic, ...
    root_path = "../"
    file_name = root_path + "StrIDR_database.json"
    # Load the JSON data from the file
    with open(file_name, 'r') as file:
        data = json.load(file)
    pdbs = data.keys()
    #for dataset in ["CLUST_CONTACT", "CLUST_EPPIC", "CLUST_PISA"]:
    for dataset in ["CONTACT","CLUST_EPPIC", "CLUST_PISA", "CLUST_CONTACT"]:
        print(dataset)
        with open(f"{root_path}data_collection/cv_splits/{dataset}/combined.txt", "r") as f:
            # read lines but remove \n
            lines = f.readlines()
            prot_pairs = [line.strip() for line in lines]
        with open(f"{root_path}data_collection/cv_splits/{dataset}/combined_info.txt", "r") as f:
            lines_info = f.readlines()
            prot_pairs_info = [line.strip() for line in lines_info]

        path_to_data = f"{root_path}data_collection/cv_splits/"
        path_to_pdb = f"{root_path}PInet/data/exon/pdb/"
        path_to_map = f"{root_path}data_collection/uniprot_EnsemblExonPDB_IDRs/"

        filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
        df_pos = pd.read_csv(filename_pos, sep='\t')

        interacting_pairs = {"fix":[], "IDR":[], "name":[]}
        non_interacting_pairs = {"fix":[], "IDR":[], "name":[]}
        
        poss_dssp = {}
        negs_dssp = {}

        for i in tqdm(range(len(prot_pairs))):
            # check if there are disordered regions for the protein
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
            residues1, residues1_types = get_residues_name(pdb_file1)
            residues2, residues2_types = get_residues_name(pdb_file2)

            df1 = pd.read_csv(map_file1, sep='\t')
            df2 = pd.read_csv(map_file2, sep='\t')

            # remove all rows where the last column is "-"
            df1 = df1[df1['PDBResNumCIF'] != "-"]
            df2 = df2[df2['PDBResNumCIF'] != "-"]

            exons1 = set(df1['EXON'])
            exons2 = set(df2['EXON'])

            # Define the mapping for column[3] to the index in the list
            # for DSSP
            mapping = {"alpha": 0, "beta": 1, "N": 2, "non": 3}
            dssp1 = {}
            dssp2 = {}
            #SecondaryStructure
            # Process each line of data
            for columns in df1.values:
                dssp1.setdefault(str(columns[0]), [0, 0, 0, 0])[mapping.get(columns[3], 0)] += 1

            for columns in df2.values:
                dssp2.setdefault(str(columns[0]), [0, 0, 0, 0])[mapping.get(columns[3], 0)] += 1
            
            for exon1 in exons1:
                for exon2 in exons2:
                    # from df1 and df2 get the column "Disorder_in_str" foi each exon
                    dis1 = df1.loc[df1['EXON'] == exon1, 'Disorder_in_str'].values
                    dis2 = df2.loc[df2['EXON'] == exon2, 'Disorder_in_str'].values
                    # exchange all "-" with "0"
                    dis1 = [0 if d == "-" else d for d in dis1]
                    dis2 = [0 if d == "-" else d for d in dis2]
                    # make dis1 and dis2 to int values
                    dis1 = [int(d) for d in dis1]
                    dis2 = [int(d) for d in dis2]
                    total_dis = np.sum(dis1) + np.sum(dis2)
                    norm_dis = total_dis / (len(dis1) + len(dis2))
                    # check if exon1 and exon2 are interacting in positive dataset
                    if (exon1, exon2, p1, p2) in zip(df_pos['EXON1'], df_pos['EXON2'], df_pos['CHAIN1'], df_pos['CHAIN2']) or \
                        (exon2, exon1, p2, p1) in zip(df_pos['EXON1'], df_pos['EXON2'], df_pos['CHAIN1'], df_pos['CHAIN2']):
                        interacting_pairs["fix"].append(1-norm_dis)
                        interacting_pairs["IDR"].append(norm_dis)
                        interacting_pairs["name"].append(tuple((exon1, exon2, prot, c1, c2, p1_info, p2_info)))
                        poss_dssp[(exon1, exon2, prot, c1, c2, p1_info, p2_info)] = [dssp1[exon1][0] + dssp2[exon2][0], dssp1[exon1][1] + dssp2[exon2][1], dssp1[exon1][2] + dssp2[exon2][2], dssp1[exon1][3] + dssp2[exon2][3]]
                    # analysis for non interacting exon pairs
                    else:
                        non_interacting_pairs["fix"].append(1-norm_dis)
                        non_interacting_pairs["IDR"].append(norm_dis)
                        non_interacting_pairs["name"].append(tuple((exon1, exon2, prot, c1, c2, p1_info, p2_info)))
                        negs_dssp[(exon1, exon2, prot, c1, c2, p1_info, p2_info)] = [dssp1[exon1][0] + dssp2[exon2][0], dssp1[exon1][1] + dssp2[exon2][1], dssp1[exon1][2] + dssp2[exon2][2], dssp1[exon1][3] + dssp2[exon2][3]]
        # save the two dicts to a file
        np.savez(f"{dataset}_IDR_pos_names_a.npy", **interacting_pairs)
        np.savez(f"{dataset}_IDR_neg_names_a.npy", **non_interacting_pairs)

        print(dataset)
        print(len(interacting_pairs["fix"]))
        print(len(non_interacting_pairs["fix"]))

        alpha_data = []
        beta_data = []
        n_data = []
        labels = []


        for key in poss_dssp.keys():
            poss_dssp[key] = poss_dssp[key][:2] + [poss_dssp[key][2] + poss_dssp[key][3]]

            alpha_data.append(poss_dssp[key][0])
            beta_data.append(poss_dssp[key][1])
            n_data.append(poss_dssp[key][2])
            labels.append('Interacting')

        print(len(alpha_data))
        print(len(beta_data))
        print(len(n_data))
        tmp_len = len(alpha_data)
        for key in negs_dssp.keys():
            negs_dssp[key] = negs_dssp[key][:2] + [negs_dssp[key][2] + negs_dssp[key][3]]

            alpha_data.append(negs_dssp[key][0])
            beta_data.append(negs_dssp[key][1])
            n_data.append(negs_dssp[key][2])
            labels.append('Non-interacting')
        print(len(alpha_data))
        print(len(beta_data))
        print(len(n_data))
        print("Non ", len(alpha_data) - tmp_len)
        
        df = pd.DataFrame({
            'Alpha': alpha_data,
            'Beta': beta_data,
            'N': n_data,
            'Exon pairs': labels
        })

    # save df
        df.to_csv(f"{dataset}_a_b_n_test.csv", index=False)

        print("Done")




    
