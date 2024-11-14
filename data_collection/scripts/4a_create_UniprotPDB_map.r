##############################################################################################
# Purpose: for each protein, create a mapping file of sequence positions in pdb and uniprot
##############################################################################################

rm(list=ls())
library(data.table)
library(gtools)
library(XML)
library(dplyr)
# turning warnings into errors
options(warn=0)

#################################################
# change datapaths here
store_dir <- '../'
process_file <- paste0(store_dir, "processed_complex_final.txt")
sifts_dir <- paste0(store_dir, 'SIFTS/')
#################################################
# all filtered uniprot pdb map
uniprot_pdb1 <- fread(process_file, sep='\t', header=TRUE)
pdbids1 <- unique(tolower(uniprot_pdb1$ID))

# chainid from uniprot matches the PDB dbChainId (i.e., the author chain id), which can be 
# different from the corresponding entity.
# The entity corresponds to chain identity in the cif file
########################################################################################
## create uniprot and pdb map based on xmls
store_dir <- paste0(store_dir, 'uniprotPDB_map')
if(!dir.exists(store_dir)){
	dir.create(store_dir)
}

for(k in 1:length(pdbids1)){ 

	# get the sub dataframe for this pdb
	tempdata <- uniprot_pdb1[uniprot_pdb1$ID == pdbids1[k], ]

	temp <- xmlParse(paste0(sifts_dir,pdbids1[k],'.xml'))

	residue_set <- getNodeSet(temp, "//rs:residue[@dbSource='PDBe']", "rs")

	PDBeResNum <- c()
	uniprotId <- c()
	chainId <- c()
	uniprotResId <- c()
	uniprotResNum <- c()

	for(j in 1:length(residue_set)){

		tempr <- xmlToList(residue_set[[j]])

		wh <- which(names(tempr) == 'crossRefDb')
		temprr <- tempr[wh]

		# take names of the dbsource
		tempdbs <- unlist(unname(lapply(temprr, function(x) unname(x['dbSource']))))

		# check which one has 'UniProt'
		whu <- which(tempdbs == 'UniProt')

		if(length(whu) != 0){

			# check whether PDB resolution is mapped
			whp <- which(tempdbs == 'PDB')
			mappedNum <- unname(temprr[[whp]]['dbResNum'])

			# if some pdb residue number is mapped
			if(mappedNum != 'null'){

				# store author chain id
				chainId <- c(chainId, temprr[[whp]]['dbChainId'])

				# Store author residue number
				PDBeResNum <- c(PDBeResNum, mappedNum)

				# UniProt
				uniprotId <- c(uniprotId, temprr[[whu]]['dbAccessionId'])
				uniprotResNum <- c(uniprotResNum, temprr[[whu]]['dbResNum'])
				uniprotResId <- c(uniprotResId, temprr[[whu]]['dbResName'])

			}

		}

	}

	chainids1 <- tempdata$CHAIN1
	chainids2 <- tempdata$CHAIN2

	unids1 <- tempdata$PROTEIN1
	unids2 <- tempdata$PROTEIN2

	for(j in 1:length(unids1)){ # for each complex in this pdbid

		# which chain to keep
		whc <- which(chainId == chainids1[j])
		uniprotId1 <- unname(uniprotId[whc])
		uniprotResNum1 <- unname(uniprotResNum[whc])
		uniprotResId1 <- unname(uniprotResId[whc])
		PDBeResNum1 <- unname(PDBeResNum[whc])
		# which uniprot to keep
		whc <- which(uniprotId1 == unids1[j])
		uniprotId1 <- unname(uniprotId1[whc])
		uniprotResNum1 <- unname(uniprotResNum1[whc])
		uniprotResId1 <- unname(uniprotResId1[whc])
		PDBeResNum1 <- unname(PDBeResNum1[whc])
		# save file
		Data1 <- data.frame(UNIPROT_SEQ_NUM=uniprotResNum1, UNIPROT=uniprotResId1,PDBResNumAuthor=PDBeResNum1)
		fwrite(Data1, paste0(store_dir,'/',paste0(unids1[j],'_',pdbids1[k],'_',chainids1[j]),'.txt'), sep='\t', quote=FALSE, row.names=FALSE)


		# which chain to keep
		whc <- which(chainId == chainids2[j])
		uniprotId2 <- unname(uniprotId[whc])
		uniprotResNum2 <- unname(uniprotResNum[whc])
		uniprotResId2 <- unname(uniprotResId[whc])
		PDBeResNum2 <- unname(PDBeResNum[whc])
		# which uniprot to keep
		whc <- which(uniprotId2 == unids2[j])
		uniprotId2 <- unname(uniprotId2[whc])
		uniprotResNum2 <- unname(uniprotResNum2[whc])
		uniprotResId2 <- unname(uniprotResId2[whc])
		PDBeResNum2 <- unname(PDBeResNum2[whc])
		# save file
		Data2 <- data.frame(UNIPROT_SEQ_NUM=uniprotResNum2, UNIPROT=uniprotResId2,PDBResNumAuthor=PDBeResNum2)
		fwrite(Data2, paste0(store_dir,'/',paste0(unids2[j],'_',pdbids1[k],'_',chainids2[j]),'.txt'), sep='\t', quote=FALSE, row.names=FALSE)
		
	}

	cat('Protein ', k, ' of ', length(pdbids1), ' done\n')

}

