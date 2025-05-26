#!/bin/bash

#example usage: 
# ./get_PPMax.sh ProteinMAE, can also be
# ./get_PPMax.sh PInet
# ./get_PPMax.sh GLINTER
# ./get_PPMax.sh dMaSIF

datasets=(CLUST_CONTACT)
modes=(test train)
methods=$1
folds="1,2,3,4,5"

for dataset in "${datasets[@]}"; do
    for mode in "${modes[@]}"; do
        for method in "${methods[@]}"; do
            echo "Running ppMax.py with Dataset: $dataset, Mode: $mode, Method: $method, Folds: $folds"
            python ppMax.py "$dataset" "$mode" "$method" "$folds"
        done
    done
done
