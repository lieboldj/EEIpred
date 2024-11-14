##############################################################################################
# Purpose: download all xml files of the cross mapping from SIFTS to map uniprot positions to PDB positions
##############################################################################################

rm(list=ls())
library(data.table)
library(gtools)
# turning warnings into errors
options(warn=2)

#################################################
# change datapaths here
store_dir <- '../'
process_file <- paste0(store_dir, "processed_complex_final.txt")
uniprot_all <- fread(process_file, sep='\t', header=TRUE)
allpdbs <- unique(tolower(uniprot_all$ID))
#################################################
## check which SIFTS data are already present
sifts_dir <- paste0(store_dir, 'SIFTS')
if(!dir.exists(sifts_dir)){dir.create(sifts_dir)}
allfiles <- list.files(sifts_dir)

allpresent <- unlist(lapply(strsplit(allfiles, '[.]'), '[[', 1))
todownload <- setdiff(allpdbs, allpresent)

## download SIFT data ----
allsifts <- substr(todownload, 2,3)

for(k in 1:length(todownload)){

	output_name <- paste0(todownload[k],'.xml.gz')
	query <- paste0('https://ftp.ebi.ac.uk/pub/databases/msd/sifts/split_xml/',allsifts[k],'/',output_name)
	cmd1 <- paste0('wget -O ',sifts_dir,'/',output_name,' ',query)
	cmd2 <- paste0("gunzip --force ", sifts_dir,'/',output_name)
	system(cmd1)
	system(cmd2)

}
