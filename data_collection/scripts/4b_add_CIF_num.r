##############################################################################################
# Purpose: add CIF number for PDB files
##############################################################################################

rm(list=ls())
library(data.table)
library(stringr)
# turning warnings into errors
options(warn=2)
#################################################
# change datapaths here
store_dir <- '../'
cif_folder <-  paste0(store_dir, 'PDB_CIF')

#################################################
# all filtered uniprot pdb map
uniprot_pdb_map_dir <- paste0(store_dir, 'uniprotPDB_map')
allfiles1 <- list.files(uniprot_pdb_map_dir, full.names=TRUE)

store_dir <- paste0(uniprot_pdb_map_dir, '_final')

if(!dir.exists(store_dir)){dir.create(store_dir)}
## Map CIF numbers ############
#####################################################################

allfiles <- list.files(cif_folder, full.names=TRUE)

for(k in 1:length(allfiles1)){
	# check if store_dir,'/',basename(allfiles1[k])) exists
	if(file.exists(paste0(store_dir,'/',basename(allfiles1[k])))){
		#cat('PDB', k, 'of', length(allfiles1), 'already done\n')
		next
	}

	if((k==810) | (k==930)| (k==1320)| (k==1321)| (k==1385)| (k==1386)| (k==2914)| (k==3908)){next}# 
	# skipped because allmap dataframe has duplicates because of "letter-code" of PDB author being missed by CIF formatting
	##-- if the dataset is created again, the above line might not be valid --> comment it out

	temp <- strsplit(basename(allfiles1[k]),'[_]')[[1]]
	temp_uni <- temp[1]
	temp_pdb <- temp[2]
	temp_chain <- strsplit(temp[3],'[.]')[[1]][1]

	wh <- which(allfiles %like% temp_pdb)
	tfile <- readLines(allfiles[wh])

	# Extract start and end positions of the coordinates entries
	wh <- which(tfile == "loop_")+1
	tfile0 <- trimws(tfile[wh])
	whh1 <- which(tfile0 == "_atom_site.group_PDB")
	start <- wh[whh1]
	#end <- wh[whh1+1]-1-2

	# if end is.na then end = length(tfile)-1
	if(whh1 == length(wh)){
	end <- length(tfile)
	}else{
	end <- wh[whh1+1]-1-2
	}

	tfile <- tfile[start:end]
	lineID <- word(tfile, 1)
	wh <- which(lineID == "ATOM")

	# Extract the field entries
	whf <- setdiff(seq(1,length(tfile)), wh)
	fields <- trimws(tfile[whf])

	tfile1 <- trimws(tfile[wh])
	tfile2 <- read.table(textConnection(tfile1), sep='', colClasses = "character")

	# author seq num
	auth_seq <- which(fields == "_atom_site.auth_seq_id")

	# cif seq num
	cif_seq <- which(fields == "_atom_site.label_seq_id")


	#filter using chain
	chainPosition <- which(fields == "_atom_site.auth_asym_id") # author-based chain ID
	chain <- tfile2[[chainPosition]]

	chain[is.na(chain)] <- "NA"
	wh <- which(chain == temp_chain)
	tfile3 <- tfile2[wh, ]
	tfile4 <- unique(tfile3[, c(cif_seq,auth_seq)])


	# map to the mapping file
	if(file.info(allfiles1[k])$size != 0){ 

		allmap <- fread(allfiles1[k])
		toadd <- rep('-', length(allmap[[1]]))

		# to catch cases where PDBResNum has a character but that is not always present in the CIF file
		tempnum <- unlist(str_extract_all(as.character(allmap$PDBResNumAuthor), "[0-9]+"))

		# to catch negative numbers
		whneg <- which(substr(as.character(allmap$PDBResNumAuthor), 1, 1) == '-')
		if(length(whneg) != 0){
			tempnum[whneg] <- paste0('-',tempnum[whneg])
		}

		whi <- intersect(tempnum, tfile4[[2]])
		wh1 <- which(tempnum %in% whi)
		wh2 <- which(tfile4[[2]] %in% whi)
		# problem with the mapping file if end does not exist
		# if wh1 1= wh2 then the mapping file is not correct
		if (length(wh1) != length(wh2)){
			cat('PDB', k, 'of', length(allfiles1), allfiles1[k],'not done\n')
			next
		}
		toadd[wh1] <- tfile4[[1]][wh2]

		allmap$PDBResNumCIF <- toadd
		fwrite(allmap, paste0(store_dir,'/',basename(allfiles1[k])), sep='\t', quote=FALSE, row.names=FALSE)
	}
	cat('PDB', k, 'of', length(allfiles1), 'done\n')

}


