#!/bin/bash

# set variables
gpu="0"
datasets=(BioGRID)
ckpt="./checkpoint/ckpt-last.pth"
folds=(1)
# Test ProteinMAE
for dataset in "${datasets[@]}"; do
    for fold in "${folds[@]}"; do
        echo "Running test_search_exon.py with Dataset: $dataset, Fold: $fold"
        CUDA_VISIBLE_DEVICES=$gpu python test_search_exon_af.py --fold $fold --ds_type $dataset --device cuda:$gpu --checkpoint ./models/TS_32 --mode test --pdb_dir "../../dmasif/surface_data/raw/01-af_pdbs/"
    done
done
