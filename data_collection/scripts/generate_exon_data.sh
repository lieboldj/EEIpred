#!/bin/bash
# stop if one scripts stops with error
set -e
# Run the R scripts in order to generate exon mappings for our train/test data

Rscript 1a_preprocess_uniprot.r
Rscript 1b_preprocess_uniprot_nr.r
Rscript 2_download_sift_mapping.r

Rscript 3_download_PDBs.R 
Rscript 4a_create_UniprotPDB_map.r
Rscript 4b_add_CIF_num.r
Rscript 5_Ensembl_exon_map.r
Rscript 6_all_map.r

# keep ../uniprot_Ensembl_Exon_map_final (that is what you need)!!!

# remove build files not needed to run the method but only remove folders if you are sure that you will not need it again
# e.g.:
# rm -rf ../PDB_CIF
# rm -rf ../SIFTS
# rm -rf ../uniprot 
# rm -rf ../uniprot_sprot_HS
# rm -rf ../uniprotPDB_map
# rm -rf ../uniprotPDB_map_final
# rm -rf ../uniprot_Ensembl_Exon_map



# UP TO THIS POINT YOU GET ALL ADDITIONAL FILES NEEDED FOR RUNNING OUR METHOD

# uncomment the following if you want to generate your own test/train sets
# based on different interaction thresholds in A or number of minimum interacting
# amino acids per exon to be classified as interacting.
#Rscript 7_chain2net.r
#Rscript 8_exonexon_contact.r
#Rscript 9_save_contact_networks.r
# 10 is only needed if chain names are too long then we have to map them to one character names
#Rscript 10_preprocessCIF2PDB_chaininfo.r
#Rscript 11_generate_GTs.r

# 12 is needed as pre-processing for the dMaSIF model and already explained in git
Rscript 12_protonation.r

