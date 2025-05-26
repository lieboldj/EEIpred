#!/bin/bash

# Set variables but better find the scripts for prediction in RRI, PPDL, and PPMax
dataset="XXX"

### if an error occurs, please make sure to delete surface_data1/processed folder and run the code again
fold="1"
cuda_no="0" # define your gpu number

# Train dMaSIF for pre-processed data, change dataset name in "dMaSIF/data.py" line 284
python -W ignore -u main_training.py --experiment_name "dMaSIF_search_3layer_12_${fold}_$dataset" --batch_size 1 --embedding_layer dMaSIF --search True --device cuda:"$cuda_no" --radius 12.0 --emb_dim 16 --n_layers 3 --train_no "$fold"

# Train PInet for pre-processed data
python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold "$fold" --dataset_name "_$dataset" --cuda "$cuda_no"

# Train GLINTER for pre-processed data
python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold "$fold" --data_root ../data_collection/cv_splits/ --dataset "$dataset" --training 1

# Train ProteinMAE for pre-processed data
python train_search.py --fold "$fold" --ds_type "$dataset" --device cuda:"$cuda_no" --ckpt ./checkpoint/ckpt-last.pth

# Run dMaSIF predictions
python -W ignore -u main_inference_exon.py --experiment_name "dMaSIF_search_3layer_12_${fold}_$dataset" --batch_size 1 --embedding_layer dMaSIF --search True --device cuda:"$cuda_no" --radius 12.0 --emb_dim 16 --n_layers 3 --train_no "$fold"

# Run PInet predictions
python utils/inference_exonpairs.py --fold $fold --dataset_name $dataset --mode train --gpu 1

# Run GLINTER predictions for proteins
python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold $fold --data_root ../data_collection/cv_splits/ --dataset $dataset
# Get exon predictions from GLINTER
python scripts/from_prot_to_exons.py --fold $fold --dataset $dataset --mode train

# Run ProteinMAE predictions
python test_search_exon.py --fold $fold --ds_type $dataset --device cuda:$cuda_no --checkpoint ./models/TS_32 --mode train

