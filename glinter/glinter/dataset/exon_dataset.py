#%%
from glinter.dataset.dimer_dataset import DimerDataset
from glinter.dataset.collater import Collater
from torch.utils.data import Dataset
import torch
from pathlib import Path
import pickle

#%%
class ProtPairDataset():
    def __init__(self, args, data:str, root:str = "", pdb_path:str = "examples/PDB/",\
                  threshold:int = 6):
        '''Load the protein pair names from the given path.'''

        with open(root+data, 'r') as fh:
            self.data = fh.read().splitlines()
        self.pdb_path = pdb_path
        self.threshold = threshold
        self.args = args
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''Return the protein pair at the given index.'''
        prot_pair = self.data[idx]
        pp_data_path = self.pdb_path + f"{prot_pair[:-2]}:{prot_pair[:-4]}{prot_pair[-2:]}/"
        self.args.dimer_root = Path(pp_data_path + f"{prot_pair[:-2]}:{prot_pair[:-4]}{prot_pair[-2:]}.pkl")
        self.args.esm_root = Path(pp_data_path)

        with open(f"{pp_data_path}labels.pkl", 'rb') as fh:
            dist = pickle.load(fh)
        dist = (dist < self.threshold).type(torch.float32)

        return DimerDataset(self.args), dist, [prot_pair[-3], prot_pair[-1]]
        