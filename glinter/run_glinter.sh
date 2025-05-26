CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 1 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC --training 1
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 2 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC --training 1
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 3 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC --training 1
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 4 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC --training 1
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 5 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC --training 1

CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 1 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 2 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 3 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 4 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC
CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 5 --data_root ../data_collection/cv_splits/ --dataset CLUST_EPPIC

datasets=(CLUST_CONTACT CLUST_PISA CLUST_EPPIC)
method=$1
splits=(train) #train val test
folds=(1 2 3 4 5)

for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for fold in "${folds[@]}"; do
            echo "Running scripts/from_prot_to_exons.py with Dataset: $dataset, Split: $split, Method: $method, Fold: $fold"
            CUDA_VISIBLE_DEVICES=1 python scripts/from_prot_to_exons.py --fold "$fold" --dataset "$dataset" --mode "$split"
        done
    done
done

#CUDA_VISIBLE_DEVICES=1 python scripts/from_prot_to_exons.py --fold 1 --dataset CLUST_CONTACT --mode test
#CUDA_VISIBLE_DEVICES=1 python scripts/from_prot_to_exons.py --fold 2 --dataset CLUST_CONTACT --mode test
#CUDA_VISIBLE_DEVICES=1 python scripts/from_prot_to_exons.py --fold 3 --dataset CLUST_CONTACT --mode test
#CUDA_VISIBLE_DEVICES=1 python scripts/from_prot_to_exons.py --fold 4 --dataset CLUST_CONTACT --mode test
#CUDA_VISIBLE_DEVICES=1 python scripts/from_prot_to_exons.py --fold 5 --dataset CLUST_CONTACT --mode test
