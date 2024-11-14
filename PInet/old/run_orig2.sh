#!/bin/bash
#python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 0 --dataset_name _EPPIC --cuda 1 
#python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 1 --dataset_name _EPPIC --cuda 1 
#python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 2 --dataset_name _EPPIC --cuda 1 
#python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 3 --dataset_name _EPPIC --cuda 1 
#python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 4 --dataset_name _EPPIC --cuda 1 

#python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 1 --dataset_name _CONTACT --cuda 1
python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 2 --dataset_name _CLUST_CONTACT --cuda 1
python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 3 --dataset_name _CLUST_CONTACT --cuda 1
python utils/train_richdbd2_fixed6mmgk.py --dataset ../data/exon --fold 4 --dataset_name _CLUST_CONTACT --cuda 1

#python utils/test_dbd.py --dataset ../data/exon --fold 0 --model seg/seg_model_protein_PISA_0_26.pth --dataset_name _PISA --cuda 0
#python utils/test_dbd.py --dataset ../data/exon --fold 1 --model seg/seg_model_protein_PISA_1_35.pth --dataset_name _PISA --cuda 0
#python utils/test_dbd.py --dataset ../data/exon --fold 2 --model seg/seg_model_protein_PISA_2_20.pth --dataset_name _PISA --cuda 1
#python utils/test_dbd.py --dataset ../data/exon --fold 3 --model seg/seg_model_protein_PISA_3_23.pth --dataset_name _PISA --cuda 1
#python utils/test_dbd.py --dataset ../data/exon --fold 4 --model seg/seg_model_protein_PISA_4_29.pth --dataset_name _PISA --cuda 1