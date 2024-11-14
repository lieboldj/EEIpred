import os
import torch
from Bio.PDB.PDBParser import PDBParser
from datasets.protein_dataset import Protein_search_exon
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config.Arguments import parser
import  numpy as np
from pathlib import Path
from models.dmasif import *

from scipy.spatial.distance import cdist
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

def save_residue_max_per_exon(_protein_pair_id, _outputs1, _outputs2, _xyz1, _xyz2, args, _proteins, exon_dir):

    extra = ""
    # own pretrained model
    if "pretrained" in args.checkpoint:
        extra = "pretrained_"
    # provided pretrained model by authors of ProteinMAE
    elif "pre" in args.checkpoint:
        extra = "pre_"
    # check if folder for results exists
    if not os.path.exists(f"results/{extra}{args.ds_type}/fold{args.fold}/{args.mode}/"):
        os.makedirs(f"results/{extra}{args.ds_type}/fold{args.fold}/{args.mode}")
    for i in range(len(_protein_pair_id)):
        # choose index for each input except exon_dir, args, and mode
        protein_pair_id = _protein_pair_id[i]
        outputs1 = _outputs1[i]
        outputs2 = _outputs2[i]
        xyz1 = _xyz1[i]
        xyz2 = _xyz2[i]
        proteins = _proteins[i]

        pdb_dir = Path(args.pdb_dir)
        protein_pair_id = protein_pair_id.split("_")
        pdb_id1 = protein_pair_id[0] + "_" + protein_pair_id[1]
        pdb_id2 = protein_pair_id[0] + "_" + protein_pair_id[2]

        proteins = proteins.split("\t")
        protein1 = proteins[0]
        protein2 = proteins[1]

        coord1 = xyz1.cpu().numpy()
        coord2 = xyz2.cpu().numpy()
        embedding1 = outputs1.cpu().numpy()
        embedding2 = outputs2.cpu().numpy()
        if np.any(np.isnan(embedding1)) or np.any(np.isnan(embedding2)):
            print(pdb_id1, pdb_id2)
            print(embedding1)
            print(embedding2)
        # pdb_id1
        parser=PDBParser(PERMISSIVE=1)
        structure=parser.get_structure("structure", f"{pdb_dir}/{pdb_id1}.pdb")
        atom_coords1 = np.stack([atom.get_coord() for atom in structure.get_atoms()])

        # pdb_id2 # remove this when not testing and take coord2 and embedding2 from running model
        parser=PDBParser(PERMISSIVE=1)
        structure2=parser.get_structure("structure", f"{pdb_dir}/{pdb_id2}.pdb")
        atom_coords2 = np.stack([atom.get_coord() for atom in structure2.get_atoms()])
    
        # from atoms to points for protein 1
        dists1 = cdist(atom_coords1, coord1)
        nn_ind1 = np.argmin(dists1, axis=1) # get closest atom

        # get embedding for all atoms from points 
        atom_emb1 = embedding1[nn_ind1]

        # from atoms to points for protein 2
        dists2 = cdist(atom_coords2, coord2)
        nn_ind2 = np.argmin(dists2, axis=1) # get closest atom

        # get embedding for all atoms from points 
        atom_emb2 = embedding2[nn_ind2]

        # predicted scores between points
        points_dists = np.matmul(atom_emb1,atom_emb2.T)

        residues1 = [residue.id[1] for residue in structure.get_residues() for atom in residue.get_atoms()]
        residues2 = [residue.id[1] for residue in structure2.get_residues() for atom in residue.get_atoms()]

        #########################################    
        residue_1 = defaultdict(list)
        for i, res in enumerate(residues1):
            residue_1[res].append(i)

        residue_2 = defaultdict(list)
        for i, res in enumerate(residues2):
            residue_2[res].append(i)

        max_values = {}
        for key1, value1 in residue_1.items():
            for key2, value2 in residue_2.items():
                max_values[key1, key2] = np.max(points_dists[np.ix_(value1, value2)])

        #########################################

        # load exon1 information
        if os.path.exists(f"{exon_dir}/{protein1}_{pdb_id1}.txt"):
            data1 = np.genfromtxt(f"{exon_dir}/{protein1}_{pdb_id1}.txt", delimiter="\t", dtype=str, skip_header=1)
        else:
            print(f"{exon_dir}/{protein1}_{pdb_id1}.txt")
            print("exon mapping files missing")

        # create dict which residue belongs to which exon from 1
        exon_dict1 = {int(line[-1]): line[0] for line in data1 if line[-1] != "-"}
        exons1 = {v: [k for k, val in exon_dict1.items() if val == v] for v in set(exon_dict1.values())}

        # load exon2 information
        if os.path.exists(f"{exon_dir}/{protein2}_{pdb_id2}.txt"):
            data2 = np.genfromtxt(f"{exon_dir}/{protein2}_{pdb_id2}.txt", delimiter="\t", dtype=str, skip_header=1)
        else:
            print("exon mapping files missing")

        # create dict which residue belongs to which exon from 2
        exon_dict2 = {int(line[-1]): line[0] for line in data2 if line[-1] != "-"}
        exons2 = {v: [k for k, val in exon_dict2.items() if val == v] for v in set(exon_dict2.values())}

        # create matrix for each exon pair and save it
        exon_find = False
        #pdb_id2 = pdb_id2.split("_")[1]
        for ex1 in exons1:
            for ex2 in exons2:
                exons1_indices = exons1[ex1]
                exons2_indices = exons2[ex2]
                exons_values = []
                for ex1_index in exons1_indices:
                    for ex2_index in exons2_indices:
                        key = (ex1_index, ex2_index)
                        exons_values.append(max_values.get(key, 0))
                matrix = np.array(exons_values).reshape(len(exons1_indices), len(exons2_indices))
                used = matrix.shape[0] <= 100 and matrix.shape[1] <= 100
                if used:
                    exon_find = True
                    np.save(f"results/{extra}{args.ds_type}/fold{args.fold}/{args.mode}/{protein1}_{protein2}_{pdb_id1}_{pdb_id2}_{ex1}_{ex2}.npy", matrix)

                else:
                    exon_find = True
                    np.save(f"results/{extra}{args.ds_type}/fold{args.fold}/{args.mode}/{protein1}_{protein2}_{pdb_id1}_{pdb_id2}_{ex1}_{ex2}_big.npy", matrix)

        if not exon_find:
            print(protein1, protein2, pdb_id1, pdb_id2)

def iterate(
    net,
    dataset,
    args,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    net.eval()
    torch.set_grad_enabled(False)

    total_processed_pairs = 0
    # Loop over one epoch:
    for it, (xyz1, normal1, label1, curvature1, dist1, atom_type1, xyz2, normal2, label2, curvature2, dist2, atom_type2, pdb_pair, proteins) in enumerate(
        tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):

        total_processed_pairs += len(xyz1)
        xyz1 = xyz1.to(args.device)
        normal1 = normal1.to(args.device)
        label1 = label1.to(args.device)
        curvature1 = curvature1.to(args.device)
        dist1 = dist1.to(args.device)
        atom_type1 = atom_type1.to(args.device)
        xyz2 = xyz2.to(args.device)
        normal2 = normal2.to(args.device)
        label2 = label2.to(args.device)
        curvature2 = curvature2.to(args.device)
        dist2 = dist2.to(args.device)
        atom_type2 = atom_type2.to(args.device)

        outputs1 = net(xyz1, normal1, curvature1, dist1, atom_type1)
        outputs2 = net(xyz2, normal2, curvature2, dist2, atom_type2)

        if args.save_exons:
            save_residue_max_per_exon(pdb_pair, outputs1, outputs2, xyz1, xyz2, args, proteins, Path(args.exon_dir))

if __name__ == "__main__":
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    net = dMaSIF(args)
    if args.checkpoint:
        ckpt_path = args.checkpoint + "_" + str(args.ds_type) + "_" + str(args.fold)
        weight = torch.load(ckpt_path)['model_state_dict']
        net.load_state_dict(weight)

    net = net.to(args.device)
    if args.mode == "train":
        trainset = Protein_search_exon(phase='train', rot_aug = False, sample_type = 'uniform', sample_num = args.downsample_points, ds_type=args.ds_type, fold=args.fold)
        dataloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers = 0
    )
    elif args.mode == "test":
        testset = Protein_search_exon(phase='test', rot_aug = False, sample_type = 'uniform', sample_num = args.downsample_points, ds_type=args.ds_type, fold=args.fold)
        dataloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=0
        )   
 
    # Perform one pass through the data:

    # check whether there is pretrained in checkpoint name
    extra = ""
    if "pretrained" in args.checkpoint:
        extra = "pretrained_"

    if args.save_exons: 
        if not os.path.exists(f"results/{extra}{args.ds_type}/fold{args.fold}/{args.mode}"):
            os.makedirs(f"results/{extra}{args.ds_type}/fold{args.fold}/{args.mode}")

    iterate(
        net,
        dataloader,
        args,
    )




















