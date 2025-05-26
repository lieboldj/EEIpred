#!/bin/bash

# set variables
gpu="0"
datasets=(CLUST_CONTACT CLUST_EPPIC CLUST_PISA)
ckpt="./checkpoint/ckpt-last.pth"
folds=(1 2 3 4 5)

# Train ProteinMAE
for dataset in "${datasets[@]}"; do
    for fold in "${folds[@]}"; do
        echo "Running train_search.py with Dataset: $dataset, Fold: $fold"
        python train_search.py --fold $fold --ds_type $dataset --device cuda:$gpu --ckpt $ckpt
    done
done
# remove the epochXX from the highest epoch to test
# Test ProteinMAE
for dataset in "${datasets[@]}"; do
   for fold in "${folds[@]}"; do
       echo "Running test_search_exon.py with Dataset: $dataset, Fold: $fold"
       CUDA_VISIBLE_DEVICES=$gpu python test_search_exon.py --fold $fold --ds_type $dataset --device cuda:$gpu --checkpoint ./models/TS_32 --mode train
       CUDA_VISIBLE_DEVICES=$gpu python test_search_exon.py --fold $fold --ds_type $dataset --device cuda:$gpu --checkpoint ./models/TS_32 --mode val
       CUDA_VISIBLE_DEVICES=$gpu python test_search_exon.py --fold $fold --ds_type $dataset --device cuda:$gpu --checkpoint ./models/TS_32 --mode test
   done
done
