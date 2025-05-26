import torch
import numpy as np
from helper import *
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time
import os

def process_single(protein_pair, chain_idx=1):
    """Turn the PyG data object into a dict."""

    P = {}
    with_mesh = "face_p1" in protein_pair.keys()
    preprocessed = "gen_xyz_p1" in protein_pair.keys()

    if chain_idx == 1:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p1 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p1_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p1 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p1 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p1
        P["batch_atoms"] = protein_pair.atom_coords_p1_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p1
        P["atomtypes"] = protein_pair.atom_types_p1

        P["xyz"] = protein_pair.gen_xyz_p1 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p1 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p1 if preprocessed else None
        P["labels"] = None #protein_pair.gen_labels_p1 if preprocessed else None

    elif chain_idx == 2:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p2 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p2_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p2 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p2 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p2
        P["batch_atoms"] = protein_pair.atom_coords_p2_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p2
        P["atomtypes"] = protein_pair.atom_types_p2

        P["xyz"] = protein_pair.gen_xyz_p2 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p2 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p2 if preprocessed else None
        P["labels"] = None #protein_pair.gen_labels_p2 if preprocessed else None

    return P

def process(args, protein_pair, net):
    P1 = process_single(protein_pair, chain_idx=1)
    if not "gen_xyz_p1" in protein_pair.keys():
        net.preprocess_surface(P1)
    P2 = None
    if not args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2)
        if not "gen_xyz_p2" in protein_pair.keys():
            net.preprocess_surface(P2)

    return P1, P2


def generate_matchinglabels(args, P1, P2):
    if args.random_rotation:
        P1["xyz"] = torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
        P2["xyz"] = torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
    xyz1_i = LazyTensor(P1["xyz"][:, None, :].contiguous())
    xyz2_j = LazyTensor(P2["xyz"][None, :, :].contiguous())

    xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1).sqrt()
    xyz_dists = (1.0 - xyz_dists).step()

    p1_iface_labels = (xyz_dists.sum(1) > 1.0).float().view(-1)
    p2_iface_labels = (xyz_dists.sum(0) > 1.0).float().view(-1)

    P1["labels"] = p1_iface_labels
    P2["labels"] = p2_iface_labels


def extract_single(P_batch, number):
    P = {}  # First and second proteins
    batch = P_batch["batch"] == number
    batch_atoms = P_batch["batch_atoms"] == number

    with_mesh = P_batch["labels"] is not None
    # Ground truth labels are available on mesh vertices:
    P["labels"] = P_batch["labels"][batch] if with_mesh else None

    P["batch"] = P_batch["batch"][batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][batch]
    P["normals"] = P_batch["normals"][batch]

    # Atom information:
    P["atoms"] = P_batch["atoms"][batch_atoms]
    P["batch_atoms"] = P_batch["batch_atoms"][batch_atoms]

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = P_batch["atom_xyz"][batch_atoms]
    P["atomtypes"] = P_batch["atomtypes"][batch_atoms]

    return P


def iterate_protmae(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None,
    pdb_ids=None,
    summary_writer=None,
    epoch_number=None,
    test_info=None,
    exon_dir=None,
    pdb_dir=None,
    dataset_name=None,
    mode="test",
    datatype="pdb"
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    total_prots = 0

    # Loop over one epoch:
    for it, protein_pair in enumerate(
        tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):
        per_pair_start = time.time()
        protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1

        batch_ids = pdb_ids[
                total_processed_pairs : total_processed_pairs + protein_batch_size
            ]
        total_processed_pairs += protein_batch_size
        # get corresponding protein names for exon mapping
        protein_pair_names = test_info[
            total_prots : total_prots + protein_batch_size]
        total_prots += protein_batch_size

        # process protein names given in *info* file
        #protein_names = protein_pair_names[0].split("\t")
        #protein1 = protein_names[0]
        #protein2 = protein_names[1]
        protein_pair.to(args.device)

        if not test:
            optimizer.zero_grad()

        # Generate the surface:
        torch.cuda.synchronize()
        surface_time = time.time()
        P1_batch, P2_batch = process(args, protein_pair, net)
        torch.cuda.synchronize()
        surface_time = time.time() - surface_time

        for protein_it in range(protein_batch_size):

            torch.cuda.synchronize()
            iteration_time = time.time()

            P1 = extract_single(P1_batch, protein_it)
            P2 = None if args.single_protein else extract_single(P2_batch, protein_it)
            # here save P1_batch
            # here save P2_batch

            if args.random_rotation:
                P1["rand_rot"] = protein_pair.rand_rot1.view(-1, 3, 3)[0]
                P1["atom_center"] = protein_pair.atom_center1.view(-1, 1, 3)[0]
                P1["xyz"] = P1["xyz"] - P1["atom_center"]
                P1["xyz"] = (
                    torch.matmul(P1["rand_rot"], P1["xyz"].T).T
                ).contiguous()
                P1["normals"] = (
                    torch.matmul(P1["rand_rot"], P1["normals"].T).T
                ).contiguous()
                if not args.single_protein:
                    P2["rand_rot"] = protein_pair.rand_rot2.view(-1, 3, 3)[0]
                    P2["atom_center"] = protein_pair.atom_center2.view(-1, 1, 3)[0]
                    P2["xyz"] = P2["xyz"] - P2["atom_center"]
                    P2["xyz"] = (
                        torch.matmul(P2["rand_rot"], P2["xyz"].T).T
                    ).contiguous()
                    P2["normals"] = (
                        torch.matmul(P2["rand_rot"], P2["normals"].T).T
                    ).contiguous()
            else:
                P1["rand_rot"] = torch.eye(3, device=P1["xyz"].device)
                P1["atom_center"] = torch.zeros((1, 3), device=P1["xyz"].device)
                if not args.single_protein:
                    P2["rand_rot"] = torch.eye(3, device=P2["xyz"].device)
                    P2["atom_center"] = torch.zeros((1, 3), device=P2["xyz"].device)   
            torch.cuda.synchronize()
            prediction_time = time.time()
            outputs = net(P1, P2)
            torch.cuda.synchronize()
            prediction_time = time.time() - prediction_time

            P1 = outputs["P1"]
            P2 = outputs["P2"]

            # from P1 and P2 input_features, right the first 10 fesature to key [curvature] and the second 6 to [chemical] from dim=1
            P1["curvature"] = P1["input_features"][:, :10]
            P1["chemical"] = P1["input_features"][:, 10:]
            P2["curvature"] = P2["input_features"][:, :10]
            P2["chemical"] = P2["input_features"][:, 10:]

            # save in one variable protein_pair the protein1 and protein2
            protein_pair_dict = dict()
            # only keep xyz, normals, atom_coords, atom_types, curvature and chemical
            key_list = ["xyz", "normals", "atoms", "atomtypes", "curvature", "chemical", "atom_center"]
            for key in P1:
                if key in key_list:
                    protein_pair_dict[key + "1"] = P1[key]
                    if not args.single_protein:
                        protein_pair_dict[key + "2"] = P2[key]

            # rename the keys and make normals1 and normals2 to normal1 and normal2
            protein_pair_dict["normal1"] = protein_pair_dict["normals1"]
            protein_pair_dict["normal2"] = protein_pair_dict["normals2"]
            del protein_pair_dict["normals1"]
            del protein_pair_dict["normals2"]
            # same with atom_types  
            protein_pair_dict["atom_type1"] = protein_pair_dict["atomtypes1"]
            protein_pair_dict["atom_type2"] = protein_pair_dict["atomtypes2"]
            del protein_pair_dict["atomtypes1"]
            del protein_pair_dict["atomtypes2"]
            # also change atom_coords to atom
            protein_pair_dict["atom1"] = protein_pair_dict["atoms1"]
            protein_pair_dict["atom2"] = protein_pair_dict["atoms2"]
            del protein_pair_dict["atoms1"]
            del protein_pair_dict["atoms2"]

            # make sure we have numpy array and no tensors for each of the keys
            for key in protein_pair_dict:
                protein_pair_dict[key] = protein_pair_dict[key].cpu().numpy()

            # split protein_pair dict and save as npy files indivially for 1 and 2 and remove the 1 or 2 from the key name
            prot_dict1 = {key[:-1]: protein_pair_dict[key] for key in protein_pair_dict}
            prot_dict2 = {key[:-1]: protein_pair_dict[key] for key in protein_pair_dict}

            # for each key also copy the values from protein_pair_dict
            for key in prot_dict1:
                prot_dict1[key] = protein_pair_dict[key + "1"]
                prot_dict2[key] = protein_pair_dict[key + "2"]
            
            del protein_pair_dict["atom_center1"]
            del protein_pair_dict["atom_center2"]

            # save P1 and P2 as npy files but keeping the dict format. 
            # create dir if not existing
            

            # save as npy files
            ######## FOR PDBs ##########
            # extract chain info from batch_ids
            if datatype == "pdb":
                if not os.path.exists(f"../data_collection/processed_dmasif/pairs/"):
                    os.makedirs(f"../data_collection/processed_dmasif/pairs/")
                np.savez(f"../data_collection/processed_dmasif/pairs/{batch_ids[protein_it]}.npz", **protein_pair_dict)

                chain1 = batch_ids[protein_it].split("_")[1]
                chain2 = batch_ids[protein_it].split("_")[2]
                base = batch_ids[protein_it].split("_")[0]

                if not os.path.exists(f"../data_collection/processed_dmasif/single/"):
                    os.makedirs(f"../data_collection/processed_dmasif/single/")
                
                np.savez(f"../data_collection/processed_dmasif/single/{base}_{chain1}.npz", **prot_dict1)
                np.savez(f"../data_collection/processed_dmasif/single/{base}_{chain2}.npz", **prot_dict2)
            ######## FOR ALPHAFOLD ##########
            if datatype == "alphafold":
                if not os.path.exists(f"../data_collection/processed_dmasif/alphafold_pairs/"):
                    os.makedirs(f"../data_collection/processed_dmasif/alphafold_pairs/")
                np.savez(f"../data_collection/processed_dmasif/alphafold_pairs/{batch_ids[protein_it]}.npz", **protein_pair_dict)

                chain1 = batch_ids[protein_it].split("\t")[0]
                chain2 = batch_ids[protein_it].split("\t")[1]

                np.savez(f"../data_collection/processed_dmasif/alphafold/{chain1}.npz", **prot_dict1)
                np.savez(f"../data_collection/processed_dmasif/alphafold/{chain2}.npz", **prot_dict2)

            #print(batch_ids[protein_it], time.time() - per_pair_start)



def iterate_surface_precompute(dataset, net, args):
    processed_dataset = []
    for it, protein_pair in enumerate(tqdm(dataset)):
        protein_pair.to(args.device)
        P1, P2 = process(args, protein_pair, net)
        if args.random_rotation:

            # get center of atoms
            atom_center1 = protein_pair.atom_coords_p1.mean(dim=-2, keepdim=True)
            atom_center2 = protein_pair.atom_coords_p2.mean(dim=-2, keepdim=True)
#
            protein_pair.atom_coords_p1 = protein_pair.atom_coords_p1 - atom_center1
            protein_pair.atom_coords_p2 = protein_pair.atom_coords_p2 - atom_center2
#
            P1["xyz"] = P1["xyz"] - atom_center1
            P2["xyz"] = P2["xyz"] - atom_center2
#
            protein_pair.atom_center1 = atom_center1
            protein_pair.atom_center2 = atom_center2
#
            # set random rotation parameters
            R1 = tensor(Rotation.random().as_matrix())
            R2 = tensor(Rotation.random().as_matrix())
#
            protein_pair.atom_coords_p1 = torch.matmul(R1, protein_pair.atom_coords_p1.T).T
            P1["xyz"] = torch.matmul(R1, P1["xyz"].T).T
            P1["normals"] = torch.matmul(R1, P1["normals"].T).T
#
            protein_pair.atom_coords_p2 = torch.matmul(R2, protein_pair.atom_coords_p2.T).T
            P2["xyz"] = torch.matmul(R2, P2["xyz"].T).T
            P2["normals"] = torch.matmul(R2, P2["normals"].T).T
#
            protein_pair.rand_rot1 = R1
            protein_pair.rand_rot2 = R2

            # perform random rotation
            P1["rand_rot"] = protein_pair.rand_rot1
            P1["atom_center"] = protein_pair.atom_center1
            P1["xyz"] = (
                torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
            )
            P1["normals"] = torch.matmul(P1["rand_rot"].T, P1["normals"].T).T
            if not args.single_protein:
                P2["rand_rot"] = protein_pair.rand_rot2
                P2["atom_center"] = protein_pair.atom_center2
                P2["xyz"] = (
                    torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
                )
                P2["normals"] = torch.matmul(P2["rand_rot"].T, P2["normals"].T).T
        protein_pair.gen_xyz_p1 = P1["xyz"]
        protein_pair.gen_normals_p1 = P1["normals"]
        protein_pair.gen_batch_p1 = P1["batch"]
        protein_pair.gen_labels_p1 = P1["labels"]
        protein_pair.gen_xyz_p2 = P2["xyz"]
        protein_pair.gen_normals_p2 = P2["normals"]
        protein_pair.gen_batch_p2 = P2["batch"]
        protein_pair.gen_labels_p2 = P2["labels"]
        processed_dataset.append(protein_pair.to("cpu"))
    return processed_dataset
