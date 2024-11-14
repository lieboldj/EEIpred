##############################################################################################
# Purpose: create exon mapping for each of the uniprot-Ensembl-pdb mapping
##############################################################################################

rm(list=ls())
library(data.table)
library(seqinr)
library(stringr)
library(plyr)
# turning warnings into errors
options(warn=2)
########################################################################################
# change datapaths here
data_dir <- "../"
process_file <- paste0(data_dir, "processed_complex_final.txt")
uniprot_ensembl <- paste0(data_dir, "uniprot_Ensembl_Exon_map")
uniprot_pdb_dir <- paste0(data_dir, "uniprotPDB_map_final")
pdb_ids_file <- paste0(data_dir, "allPDB_IDs.txt")
########################################################################################
uniprot_pdb <- data.table::fread(process_file, sep='\t', header=TRUE)
# all uniprot-exon mapped files
allfiles1 <- list.files(uniprot_ensembl, full.names=TRUE)

# all uniprot-pdb mapped files
allfiles2 <- list.files(uniprot_pdb_dir, full.names=TRUE)

# store the mappings
store_dir <- paste0(data_dir, "uniprot_EnsemblExonPDB_map")
unlink(store_dir, recursive=TRUE)
if(!dir.exists(store_dir)){
	dir.create(store_dir)
}

# store not mapped files
nomap <- c()

# for each of the genes in the uniprot_pdb_f
for(k in 1:length(allfiles1)){
	# check if store_dir,'/',basename(allfiles1[k])) exists
	if(file.exists(paste0(store_dir,'/',basename(allfiles1[k])))){
		cat('Seq ', k, 'of', length(allfiles1), 'already done\n')
		next
	}

	# uniprot exon map file
	tempmapUE <- data.table::fread(allfiles1[k],header=TRUE)

	# uniprot PDB map file
	wh <- which(allfiles2 %like% basename(allfiles1[k]))
	tempfile <- allfiles2[wh]

	if(length(tempfile) != 0){ # check whether this is present in mapped files
		tempmapUP <- data.table::fread(tempfile)
		tempmapUP <- unique(tempmapUP)
	}else{
		nomap <- c(nomap, k)
		next
		# break
	}

	# PDB author seq number entry
	temppdb <- rep('-', length(tempmapUE[[1]]))
	tempcif <- rep('-', length(tempmapUE[[1]]))

	whi <- intersect(tempmapUE$UNIPROT_SEQ_NUM, tempmapUP$UNIPROT_SEQ_NUM)
	wh <- which(tempmapUE$UNIPROT_SEQ_NUM %in% whi)
	temppdb[wh] <- tempmapUP$PDBResNumAuthor # place the pdb seq nums
	tempcif[wh] <- tempmapUP$PDBResNumCIF # place the pdb seq nums

	temp1 <- cbind(tempmapUE[,c(3,4)],tempmapUE[,c(2)])
	temp3 <- cbind(temp1,tempmapUE[,c(1)])
	colnames(temp3) <- c('EXON', 'EXON_NUM','UNIPROT','UNIPROT_SEQ_NUM')

	# final map
	temp3$PDBResNumAuthor <- temppdb
	temp3$PDBResNumCIF <- tempcif

	fwrite(temp3, paste0(store_dir,'/',basename(tempfile)), row.names=FALSE, sep='\t', quote=FALSE)

	cat('Seq ', k, 'of', length(allfiles1), 'done\n')

}

##--- save the list of pdb ids containing potential complexes
allpdbs <- unique(unlist(lapply(strsplit(basename(allfiles1), '[_]'), '[[', 2)))
data.table::fwrite(data.frame(allpdbs), pdb_ids_file, col.names=FALSE, row.names=FALSE, quote=FALSE)
