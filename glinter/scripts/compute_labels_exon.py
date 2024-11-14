import numpy as np
from Bio.PDB.PDBParser import PDBParser
import pickle
import torch
from tqdm import tqdm
import os

# load txt file for pdb pairs from data/PDBexonData.txt
with open(f"../data/PDBexonData.txt", "r") as f:
    pdb_pairs = f.read().splitlines()
counter = 0
for pdb_pair in tqdm(pdb_pairs):
    
    pdb_dir = f"../examples/PDB"
    protein_pair_id = pdb_pair

    protein_pair_id = protein_pair_id.split("_")
    pdb_id1 = protein_pair_id[0] + "_" + protein_pair_id[1]
    pdb_id2 = protein_pair_id[0] + "_" + protein_pair_id[2]

    if not os.path.exists(f"../examples/PDB/{pdb_id1}:{pdb_id2}/labels.pkl"):
        if not os.path.exists(f"../examples/PDB/{pdb_id2}:{pdb_id1}/labels.pkl"):
            counter += 1
            print("skipping", pdb_id1, pdb_id2)
            continue
    if os.path.exists(f"../examples/PDB/{pdb_id1}:{pdb_id2}/labels_residues.pkl"):
        continue
    elif os.path.exists(f"../examples/PDB/{pdb_id2}:{pdb_id1}/labels_residues.pkl"):
        continue
    # pdb_id1
    parser=PDBParser(PERMISSIVE=1)
    # Parse the PDB files and get the structures
    structure1 = parser.get_structure('structure1', f"{pdb_dir}/{pdb_id1}.pdb")
    structure2 = parser.get_structure('structure2', f"{pdb_dir}/{pdb_id2}.pdb")

    # Create a numpy array to hold the residue pair labels
    num_residues1 = len(list(structure1.get_residues()))
    num_residues2 = len(list(structure2.get_residues()))
    residue_pair_matrix = np.zeros((num_residues1, num_residues2))

    interacting = False
    # Calculate distances between all atom pairs in the two structures
    for i, residue1 in enumerate(structure1.get_residues()):
        for j, residue2 in enumerate(structure2.get_residues()):
            interacting = False
            for atom1 in residue1.get_atoms():
                for atom2 in residue2.get_atoms():
                    # this actually gives the distance in Angstrom between the two atoms
                    distance = atom1 - atom2
                    #distance = np.sqrt(((atom1.get_coord() - atom2.get_coord())**2).sum(-1))
                    if distance <= 6.0:
                        #print(atom1.get_coord(), atom2.get_coord(), distance)
                        #print(residue1.id[1], residue2.id[1], residue_pair_matrix.shape)
                        residue_pair_matrix[i, j] = 1
                        interacting = True
                        break
                if interacting:
                    break
    with open(f"../examples/PDB/{pdb_id1}:{pdb_id2}/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    labels = (labels < 6).float()
    try:
        if not np.allclose(residue_pair_matrix, labels):
            print(residue_pair_matrix.shape, labels.shape)
            print(np.sum(residue_pair_matrix), torch.sum(labels))
    except:
        print("shapes: ", residue_pair_matrix.shape, labels.shape)
    # save residue matrix to pkl file
    with open(f"../examples/PDB/{pdb_id1}:{pdb_id2}/labels_residues.pkl", "wb") as f:
        pickle.dump(residue_pair_matrix, f)
        #print("saved", pdb_id1, pdb_id2)
print(counter)