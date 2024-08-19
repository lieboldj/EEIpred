# training example
python train_search.py --fold 1 --ds_type CONTACT --device cuda:0 --ckpt ./checkpoint/ckpt-last.pth
# inference example to get RRI predictions
python test_search_exon.py --fold 1 --ds_type CONTACT --device cuda:0 --checkpoint ./models/Transformer_search_batch32_pre --mode test
python test_search_exon.py --fold 1 --ds_type CONTACT --device cuda:0 --checkpoint ./models/Transformer_search_batch32_pre --mode train

