CUDA_VISIBLE_DEVICES=1 python glinter/models/msa_model_train1.py --dimer-root x --feature heavy-atom-graph,surface-graph,coordinate-ca-graph,pickled-esm --fold 1 --data_root ../data_collection/cv_splits/ --dataset RUNTIME
python scripts/from_prot_to_exons.py --fold 1 --dataset RUNTIME --mode test
#python scripts/from_prot_to_exons.py --fold 1 --dataset CONTACT --mode train
#python scripts/from_prot_to_exons.py --fold 2 --dataset CONTACT --mode train
#python scripts/from_prot_to_exons.py --fold 3 --dataset CONTACT --mode train
#python scripts/from_prot_to_exons.py --fold 4 --dataset CONTACT --mode train
#python scripts/from_prot_to_exons.py --fold 5 --dataset CONTACT --mode train
#
#python scripts/from_prot_to_exons.py --fold 1 --dataset CONTACT --mode test
#python scripts/from_prot_to_exons.py --fold 2 --dataset CONTACT --mode test
#python scripts/from_prot_to_exons.py --fold 3 --dataset CONTACT --mode test
#python scripts/from_prot_to_exons.py --fold 4 --dataset CONTACT --mode test
#python scripts/from_prot_to_exons.py --fold 5 --dataset CONTACT --mode test

