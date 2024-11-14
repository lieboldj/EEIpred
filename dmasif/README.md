environment check, orginial tools "Link to dmasif"
get pdbs 

put pdbs with hydrogen atoms in surface_data/raw/01-benchmark_pdbs
run convert_pdb2npy.py to preprocess the files to npy format
in ./dmasif $ python convert_pdb2npy.py

if train_no = "", then it is simply taking the train.txt and test.txt

each method inference will output the predictions per residue pair on exon pair
level
get_resi_results.py will calculate the RRI results

to use 