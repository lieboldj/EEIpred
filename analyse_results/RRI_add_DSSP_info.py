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
    path_to_map = "../data_collection/uniprot_EnsemblExonPDB_map/"
    df_aa = pd.read_csv("../data_collection/aa_interactions6_DSSP.txt", sep='\t')
    # rename last two columns to "AA1" and "AA2"
    df_aa.columns = ['chain1', 'chain2', 'exon1', 'exon2', 'AA1', 'AA2', 'DSSP1', 'DSSP2']
    # add two new columns to the dataframe

    last_chain1 = ""
    last_chain2 = ""
    # iterate over each row of the dataframe
    for index, row in tqdm(df_aa.iterrows()):
        if pd.isna(row["DSSP1"]) or pd.isna(row["DSSP2"]):
            #print(row)
            # check whether the chain1 and chain2 are the same as the last chain1 and chain2
            if not(row["chain1"] == last_chain1) or not (row["chain2"] == last_chain2):
                file1 = path_to_map + row["chain1"]+".txt"
                file2 = path_to_map + row["chain2"]+".txt"
                # only load header for file1 and file2
                head1 = np.genfromtxt(file1, delimiter="\t", dtype=str, max_rows=1)
                head2 = np.genfromtxt(file2, delimiter="\t", dtype=str, max_rows=1)
                if len(head1) < 7 or len(head2) < 7:
                    continue
                data1 = np.genfromtxt(file1, delimiter="\t", dtype=str, skip_header=1)
                data2 = np.genfromtxt(file2, delimiter="\t", dtype=str, skip_header=1)
                # remove lines where line[-1] == "-"
                data1 = [line for line in data1 if line[-1] != "-"]
                data2 = [line for line in data2 if line[-1] != "-"]
                last_chain1 = row["chain1"]
                last_chain2 = row["chain2"]

            if len(head1) < 7 or len(head2) < 7:
                continue
            try:
                data1_aa_list = [line[3] for line in data1 if int(line[-3]) == row["AA1"]]
                data2_aa_list = [line[3] for line in data2 if int(line[-3]) == row["AA2"]]
            except:
                print("Error: ", row["chain1"], row["chain2"], row["AA1"], row["AA2"])
                continue

            # Directly assign the string if the list is not empty
            data1_aa = data1_aa_list[0] if data1_aa_list else ""
            data2_aa = data2_aa_list[0] if data2_aa_list else ""

            df_aa.at[index, 'DSSP1'] = data1_aa
            df_aa.at[index, 'DSSP2'] = data2_aa

    # uncomment to save information to a file
    #df_aa.to_csv("../data_collection/aa_interactions6_DSSP.txt", sep='\t', index=False)
