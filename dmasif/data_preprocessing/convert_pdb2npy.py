import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *
import subprocess
import time
ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


def convert_pdbs(pdb_dir, npy_dir):
    print("Converting PDBs")
    pdb_list = "/cosybio/project/EEIP/EEIP/data_collection/cv_splits/RUNTIME/test1.txt"
    # extract pdbs from list
    with open(pdb_list, 'r') as f:
        pdb_list = f.read().splitlines()
    for pdb_pair in pdb_list:

        pdb_split = pdb_pair.split("_")
        pdb_l = pdb_split[0] + "_" + pdb_split[1]
        pdb_r = pdb_split[0] + "_" + pdb_split[2]

        subprocess.call(f"/cosybio/project/EEIP/EEIP/glinter/external/reduce/reduce -build -Quiet -DB /cosybio/project/EEIP/EEIP/glinter/external/reduce/reduce_wwPDB_het_dict.txt /cosybio/project/EEIP/EEIP/PInet/data/exon/pdb/{pdb_l}.pdb > {pdb_dir}/reduced/{pdb_l}.pdb", shell=True)
        subprocess.call(f"/cosybio/project/EEIP/EEIP/glinter/external/reduce/reduce -build -Quiet -DB /cosybio/project/EEIP/EEIP/glinter/external/reduce/reduce_wwPDB_het_dict.txt /cosybio/project/EEIP/EEIP/PInet/data/exon/pdb/{pdb_r}.pdb > {pdb_dir}/reduced/{pdb_r}.pdb", shell=True)
        
        #reduce -Trim -Quiet -DB /cosybio/project/EEIP/EEIP/data_collection/Reduce_3.23/ -i 1a3a_A.pdb -o 1a3a_A.pdb
        
        protein_l = load_structure_np(pdb_dir / ("reduced/"+pdb_l + ".pdb"), center=False)
        protein_r = load_structure_np(pdb_dir / ("reduced/"+pdb_r + ".pdb"), center=False)
        np.save(npy_dir / (pdb_l + "_atomxyz.npy"), protein_l["xyz"])
        np.save(npy_dir / (pdb_l + "_atomtypes.npy"), protein_l["types"])
        np.save(npy_dir / (pdb_r + "_atomxyz.npy"), protein_r["xyz"])
        np.save(npy_dir / (pdb_r + "_atomtypes.npy"), protein_r["types"])
    #for p in tqdm(pdb_dir.glob("*.pdb")):
    #    protein = load_structure_np(p, center=False)
    #    np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
    #    np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])

if __name__ == "__main__":

    pdb_dir = Path("surface_data/raw/01-benchmark_pdbs/") # Path with PDB files
    npy_dir = Path("surface_data/raw/01-benchmark_surfaces_npy")

    convert_pdbs(pdb_dir, npy_dir)
    print("Done")