## Download CIF data and save fasta 

rm(list=ls())
library('seqinr')
library('data.table')
library('stringr')
# NOT turning warnings into errors because conversion of warnign during conversion of 3-letter code to AA symbol
options(warn=0)

#################################################
# change datapaths here
store_dir <- '../'
process_file <- paste0(store_dir, "processed_complex_final.txt")
cif_txt <- paste0(store_dir, 'all_CIFs.txt')
store_cif <- paste0(store_dir,'PDB_CIF') # same folder as in script 4b
################################################################
# all filtered uniprot pdb map
uniprot_all <- fread(process_file, sep='\t', header=TRUE)
keep <- unique(tolower(uniprot_all$ID))


## download CIFs ###
# check if the folder exists
if(!dir.exists(store_cif)){
  dir.create(store_cif)
}

pdbids2 <- unique(unlist(lapply(strsplit(list.files(store_cif), '[.]'), '[[', 1)))
pdbids3 <- setdiff(keep, pdbids2)
allpdbs <- paste(pdbids3, collapse=',')


if(length(pdbids3) != 0){
  writeLines(allpdbs,cif_txt)
  system(paste0('./batch_download.sh -f', cif_txt,' -o ',store_cif,' -c'))
  system(paste0('gunzip ',store_cif,'/*.cif.gz'))
}

