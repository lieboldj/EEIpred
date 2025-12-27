#!/bin/bash

# example usage:
# train to save model parameters, test to predict given the pre-trained model
# ./get_PPDL.sh dmasif train
# ./get_PPDL.sh ProteinMAE test
# ./get_PPDL.sh PInet test
# ./get_PPDL.sh GLINTER train

datasets=(CLUST_CONTACT CLUST_EPPIC CLUST_PISA)
method=$1
modes=$2
folds="1,2,3,4,5"

# only if modes is set to test, set eval_modes to test_set, val_set, train_set
if [ "$modes" = "test" ];  then
    eval_modes=(test_set val_set train_set)
    for dataset in "${datasets[@]}"; do
        for mode in "${eval_modes[@]}"; do
            echo "Running ppDL.py with Dataset: $dataset, Mode: test, Method: $method, Folds: $folds, Eval Mode: $mode"
            python ppDL.py -mth "$method" -d "$dataset" -md "test" -f "$folds" -em "$mode" 
        done
    done
else
    for dataset in "${datasets[@]}"; do
        echo "Running ppDL.py with Dataset: $dataset, Mode: train, Method: $method, Folds: $folds"
        python ppDL.py -mth "$method" -d "$dataset" -md "train" -f "$folds" -c 1
    done
fi
