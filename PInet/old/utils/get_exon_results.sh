# run inference for pre-trained model to get intermediate predictions per exon-exon for an protein protein pair
python inference_save.py --fold 0 --gpu 0 --epoch 0 --origin_path EEIP/ --exon_mapping_dir data_collection/uniprot_EnsemblExonPDB_map --database test/ --dataset_name Test_set/ --data_short_name test_pi --mode test --pdb_path pdb/ --chain_info True

python test_exon_results.py --model Maximum --index 1 --path ../seg/alex_all-samples --data "" --results ../../results/test_pi --exons ../../../data_collection/int_exon_pairs.txt
