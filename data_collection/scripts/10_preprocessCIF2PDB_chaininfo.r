##############################################################################################
# Purpose: clean the PDB files for use in the dMasif method
##############################################################################################

rm(list=ls())
library(data.table)
library(plyr)
library(ggplot2)
library(stringr)
library(igraph)
# install bio3d package
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("bio3d")
library(bio3d)

data_dir <- "../"
pdbDirectory <- paste0(data_dir,'PDB_CIF')
store_dir <- paste0(data_dir, 'PDB_chains')
if (!dir.exists(store_dir))
	dir.create(store_dir, recursive=TRUE)

dataset_dir <- paste0(data_dir, "CONTACT") # name of the dataset
cutoff <- c(6)#c(4, 5, 6, 7, 8)
# set the number of amino acids which at least have to be closer than the cutoff
num_aa <- c(1)#,3,5,7,9)
chain_path <- paste0(data_dir, 'chain_info.txt')
# Change here to dataset for int_exon_pairs.txt
pdb_inter <- fread(paste0(dataset_dir, "/int_exon_pairs",cutoff[1],".txt"), sep='\t') ## here done only for contact-based data --> chnage this input file for PISA and EPPIC to get


## preprocessed pdb files for each pdb in each contact pisa and eppic
temp_g <- graph_from_data_frame(pdb_inter[,c(1,2)], directed=FALSE)
temp_g <- simplify(temp_g, remove.multiple=TRUE, remove.loop=TRUE)
temp_d <- as.data.frame(get.edgelist(temp_g))

pdbp <- temp_d

uniprot1 <- unlist(lapply(strsplit(pdbp[[1]], '[_]'), '[[', 1))
pdbids1 <- unlist(lapply(strsplit(pdbp[[1]], '[_]'), '[[', 2))
chains1 <- unlist(lapply(strsplit(pdbp[[1]], '[_]'), '[[', 3))
pdbchains1 <- paste0(pdbids1,'_',chains1)

uniprot2 <- unlist(lapply(strsplit(pdbp[[2]], '[_]'), '[[', 1))
pdbids2 <- unlist(lapply(strsplit(pdbp[[2]], '[_]'), '[[', 2))
chains2 <- unlist(lapply(strsplit(pdbp[[2]], '[_]'), '[[', 3))
pdbchains2 <- paste0(pdbids2,'_',chains2)

allpdbchains <- union(pdbchains1, pdbchains2)

##------------- write the PDBs -------------------------
cifs <- unlist(lapply(strsplit(allpdbchains, '[_]'), '[[', 1))
allcifs <- unique(cifs)
chains <- unlist(lapply(strsplit(allpdbchains, '[_]'), '[[', 2))

newc <- c(toupper(letters), letters)
store_pdbid <- c()
org_chain <- c()
new_chain <- c()
bbflag <- 0
# to_remove <- c()

for(k in 1:length(allcifs)){

	temp_cif <- cifs[which(cifs == allcifs[k])]
	temp_chain <- chains[which(cifs == allcifs[k])]
	tfile <- readLines(paste0(pdbDirectory,"/",allcifs[k],".cif"))

	# Extract start and end positions of the coordinates entries
	wh <- which(tfile == "loop_")+1
	tfile0 <- trimws(tfile[wh])
	whh1 <- which(tfile0 == "_atom_site.group_PDB")
	start <- wh[whh1]

	#MODIFIED HERE FOR THE CORRECT END TO AVOID NaN as end
	if(whh1 == length(wh)){
		end <- length(tfile)
	}else{
		end <- wh[whh1+1]-1-2
	}

	# Extract the coordinates part of the PDB file
	tfile <- tfile[start:end]
	lineID <- word(tfile, 1)
	wh <- which(lineID == "ATOM" | lineID == "HETATM")

	# Extract the field entries
	whf <- setdiff(seq(1,max(wh)), wh)
	fields <- trimws(tfile[whf])

	tfile1 <- trimws(tfile[wh])
	tfile2 <- read.table(textConnection(tfile1), sep='', colClasses = "character")

	# take only ATOM
	tfile22 <- tfile2[which(tfile2[[1]] == "ATOM"), ]

	# extract coordinates for only the heavy atoms (S, O, C, N)
	atomPosition <- which(fields == "_atom_site.type_symbol")
	lineID2 <- tfile22[[atomPosition]]
	wh <- which(lineID2 == "C" | lineID2 == "S" | lineID2 == "O" | lineID2 == "N")
	tfile2 <- tfile22[wh, ]

	chainPosition <- which(fields == "_atom_site.auth_asym_id")#_atom_site.label_asym_id
	chain <- tfile2[[chainPosition]]
	seqPosition <- which(fields == "_atom_site.label_seq_id")#CIF based
	chain1 <- unique(temp_chain)

	for(j in 1:length(temp_chain)){

		# get positions of the chain in question
		ncc <- temp_chain[j]
		wh_chain <- which(chain == ncc)
		tfile3 <- tfile2[wh_chain,]
		tfile4 <- tfile3[,c(1,2,4,6,19,9,11,12,13,14,15,3)] 

		if(nchar(ncc) > 1){ # if the number of characters is more than one
			ncc <- setdiff(newc, chain1)[1]
			if(is.na(ncc)){
				bbflag <- 1
				break
			}
			chain1 <- union(chain1, ncc)
			store_pdbid <- c(store_pdbid, allcifs[k])
			org_chain <- c(org_chain, temp_chain[j])
			new_chain <- c(new_chain, ncc)
		}
		ncc1 <- rep(ncc, length(tfile4[[5]]))
		atm1 <- seq(1,length(tfile4[[2]]), 1)
		atms <- tfile4[[3]]
		ress <- tfile4[[4]]
		reso <- tfile4[[6]]
		hatm <- tfile4[[12]]
		oflag <- tfile4[[10]]
		bflag <- tfile4[[11]]
		coord <- tfile4[, c(7,8,9)]

		## save the duplicate entries to remove
		temp_data <- data.frame(atms,reso)
		reso1 <- unique(reso)
		to_remove <- c()
		for(i in 1:length(reso1)){
			tempd <- temp_data[temp_data$reso == reso1[i], ]$atms
			tempd1 <- count(tempd)
			temp_atms <- tempd1[[1]][which(tempd1$freq > 1)]
			if(length(temp_atms) != 0){
				for(m in 1:length(temp_atms)){
					wh1 <- which(atms == temp_atms[m])
					wh2 <- which(reso == reso1[i])
					wh <- intersect(wh1, wh2)
					to_remove <- c(to_remove, wh[2:length(wh)])
				}
			}
		}

		## remove the duplicate entries
		if(length(to_remove) != 0){
			ncc1 <- ncc1[-to_remove]
			atm1 <- atm1[-to_remove]
			atms <- atms[-to_remove]
			ress <- ress[-to_remove]
			reso <- reso[-to_remove]
			oflag <- oflag[-to_remove]
			bflag <- bflag[-to_remove]
			coord <- coord[-to_remove,]
			hatm <- hatm[-to_remove]
		}

		if(nrow(tfile4) != 0){

			write.pdb(file=paste0(store_dir,'/', allcifs[k],'_',ncc,'.pdb'),
			type=rep('ATOM', length(reso)),
			xyz=as.numeric(t(as.matrix(coord))),
			resno=reso,
			chain=ncc1,
			resid=ress,
			eleno=atm1,
			elety=atms,
			elesy=hatm,
			o=oflag,
			b=bflag
			)

		}
		
		cat('chain ',j, 'of protein', k, 'out of', length(allcifs), 'proteins done\n')
	}

	if(bbflag == 1){
		break
	}
	
}


## save the new chain data -----
allData <- data.frame(PDBID=store_pdbid, old_chain=org_chain, new_chain=new_chain)
fwrite(allData, chain_path, row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')
