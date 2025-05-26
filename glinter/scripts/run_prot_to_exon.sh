#CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train1.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 1 --data_root ../data_collection/cv_splits/ --dataset RUNTIME
#python scripts/from_prot_to_exons.py --fold 1 --dataset RUNTIME --mode test

datasets=(CLUST_CONTACT CLUST_EPPIC CLUST_PISA)
modes=(train test val)
folds=(1 2 3 4 5)

for dataset in "${datasets[@]}"; do
    for mode in "${modes[@]}"; do
        for fold in "${folds[@]}"; do
            echo "Running from_prot_to_exons.py with Dataset: $dataset, Mode: $mode, Fold: $fold"
            python scripts/from_prot_to_exons.py --fold "$fold" --dataset "$dataset" --mode "$mode"
        done
    done
done

#python scripts/from_prot_to_exons.py --fold 1 --dataset CLUST_CONTACT --mode train
