##############################################################################################
# Purpose: generate uniprot files for Homo sapiens only
# Warning: Due to huge file, the current code will take some time --> 6-7 hours. Hence, it should be run once and 
# the data should be saved
##############################################################################################

rm(list=ls())
library(seqinr)
library(data.table)
library(Rcpp)
library(stringr)
library(plyr)
# turning warnings into errors
options(warn=2)

#########################################
# change datapaths and/or species here
store_dir <- '../'
if(!dir.exists(store_dir)){dir.create(store_dir)}
data_abbr <- "HS"
species_ID_uniprot <- "Homo sapiens"
########################################

# Proteins from Uniprot...these are reviewed entries
# read uniprot data for geneid and pdb info
dat_file <- paste0(store_dir, "uniprot_sprot.dat")
print(dat_file)
if(!file.exists(dat_file)) {
	system("wget -O ../uniprot_sprot.dat.gz https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz")
	system("gunzip --force ../uniprot_sprot.dat.gz")
}
# save separate files for uniprot entries
uniprot_dir <- paste0(store_dir, 'uniprot')
if(!dir.exists(uniprot_dir)){dir.create(uniprot_dir)}

system("awk '/^ID/{if(NR!=1){for(i=0;i<j;i++)print a[i]>\"../uniprot/file\"k;j=0;k++;}a[j++]=$0;next}{a[j++]=$0;}END{for(i=0;i<j;i++)print a[i]>\"../uniprot/file\"k}' i=0 k=1  ../uniprot_sprot.dat")


# filter to only retain Homo sapiens
allfiles <- list.files(uniprot_dir, full.names=TRUE)
output_dir <- paste0(store_dir, 'uniprot_sprot_', data_abbr)
if(!dir.exists(output_dir)){dir.create(output_dir)}
counter <- 0
for(k in 1:length(allfiles)){

	tempf <- readLines(allfiles[k])

	for(j in 1:length(tempf)){

		temp <- tempf[j]
		temp2 <- substr(temp, 1,2)

		if((temp2 == 'OS') & (temp %like% species_ID_uniprot)){
			tf <- strsplit(tempf[1],'\\s+')
			filename <- paste0(output_dir,'/',tf[[1]][2],'_',strsplit(tf[[1]][3],';')[[1]][1],'.txt')
			file.copy(allfiles[k], filename)
			counter <- counter+1                            
			break
		}

	}

	cat(k,' of ', length(allfiles), ' done\n')

}
