##############################################################################################
# Purpose: get training/testing data for dMASIF, PInet, and GLINTER
##############################################################################################

rm(list=ls())
library(data.table)
library(plyr)
library(ggplot2)
library(stringr)
library(igraph)

## GT data : contact definitions -----
cutoff <- c(6)#c(4, 5, 6, 7, 8)
# set the number of amino acids which at least have to be closer than the cutoff
num_aa <- c(1)#,3,5,7,9)

gtdata <- paste0('CONTACT_net_', cutoff[1] ,'_',num_aa[1]) # if you generate data for different cutoff and num_aa, change the index accordingly
data_dir <- "../"
chain_path <- paste0(data_dir, 'chain_info.txt')
tmp_dir <- paste0(data_dir, "GT_datasets/") # name of the dataset
net_dir <- paste0(data_dir, "CONTACT_networks/") # name of the dataset
for(gt in 1:length(gtdata)){

	store_dir <- paste0(tmp_dir, gtdata[gt])
	# only create the directory if it does not exist
	if (!dir.exists(store_dir))
		dir.create(store_dir,recursive=TRUE)

	# non-interacting exon pairs
	pdb_inter <- fread(paste0(net_dir, gtdata[gt],'_negatives.txt'), sep='\t')
	wh <- union(which(pdb_inter[[1]] == ''), which(pdb_inter[[2]] == ''))
	if(length(wh) > 0){
		pdb_inter <- pdb_inter[-wh, ]
	}
	p1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 1))
	p2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 1))
	pd1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 2))
	pd2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 2))
	c1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 3))
	c2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 3))
	ch1 <- paste0(pd1,'_',c1)
	ch2 <- paste0(pd2,'_',c2)
	pdb_data <- data.frame(EXON1=pdb_inter$exon1, EXON2=pdb_inter$exon2, UNIPROT1=p1, UNIPROT2=p2, CHAIN1=ch1, CHAIN2=ch2)
	data.table::fwrite(pdb_data, paste0(store_dir,'/CONTACT_negatives.txt'), sep='\t', quote=FALSE)

	# interacting exon pairs
	pdb_inter <- fread(paste0(net_dir, gtdata[gt],'_positives.txt'), sep='\t')
	wh <- union(which(pdb_inter[[1]] == ''), which(pdb_inter[[2]] == ''))
	if(length(wh) > 0){
		pdb_inter <- pdb_inter[-wh, ]
	}
	p1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 1))
	p2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 1))
	pd1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 2))
	pd2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 2))
	c1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 3))
	c2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 3))
	ch1 <- paste0(pd1,'_',c1)
	ch2 <- paste0(pd2,'_',c2)
	pdb_data <- data.frame(EXON1=pdb_inter$exon1, EXON2=pdb_inter$exon2, UNIPROT1=p1, UNIPROT2=p2, CHAIN1=ch1, CHAIN2=ch2)


	nee <- igraph::simplify(igraph::graph_from_data_frame(pdb_data[,c(1,2)], directed=FALSE))

	# uniprot1 <- unlist(lapply(strsplit(pdb_inter[[1]], '[_]'), '[[', 1))
	pdbids1 <- unlist(lapply(strsplit(pdb_data[[5]], '[_]'), '[[', 1))
	chains1 <- unlist(lapply(strsplit(pdb_data[[5]], '[_]'), '[[', 2))

	# uniprot2 <- unlist(lapply(strsplit(pdb_inter[[2]], '[_]'), '[[', 1))
	pdbids2 <- unlist(lapply(strsplit(pdb_data[[6]], '[_]'), '[[', 1))
	chains2 <- unlist(lapply(strsplit(pdb_data[[6]], '[_]'), '[[', 2))


	#############################################################################
	# following part is for specific chain naming, you can comment if not needed
	#############################################################################
	## read the chains mapping from the previously processed data for contact_6_1
	alldx <- data.table::fread(chain_path, header=FALSE)
	store_pdbid <- alldx[[1]]
	org_chain <- alldx[[2]]
	new_chain <- alldx[[3]]

	## ---- replace the orginal chain with the new chain -----------
	for(k in 1:length(store_pdbid)){
		wh1 <- which(pdbids1 == store_pdbid[k])
		wh2 <- which(chains1 == org_chain[k])
		wh <- intersect(wh1, wh2)
		if(length(wh) != 0){
			chains1[wh] <- new_chain[k]
		}

		wh2 <- which(chains2 == org_chain[k])
		wh <- intersect(wh1, wh2)
		if(length(wh) != 0){
			chains2[wh] <- new_chain[k]
		}
	}

	pdbchains1 <- paste0(pdbids1,'_',chains1)
	pdbchains2 <- paste0(pdbids2,'_',chains2)

	##--- update pdb_data with new chains
	pdb_data$CHAIN1 <- pdbchains1
	pdb_data$CHAIN2 <- pdbchains2
	#############################################################################
	# previous part is for specific chain naming, you can comment if not needed
	#############################################################################



	# data.table::fwrite(pdb_data, paste0(store_dir,'/CONTACT_positives.txt'), sep='\t', quote=FALSE)

	##---- save list of training and testing data ---------------
	contact_data <- igraph::simplify(igraph::graph_from_data_frame(data.frame(pdbchains1, pdbchains2), directed=FALSE))

	##---- filter to keep protein pairs processed by all --------
	contact_filtered <- igraph::intersection(contact_data)#, contact_to_use)

	alldata <- data.frame(igraph::as_edgelist(contact_filtered))

	##--- get the corresponding uniprot ids ---
	uniprot1x <- c()
	uniprot2x <- c()

	for(jj in 1:length(alldata[[1]])){

		wh1 <- which(pdb_data[[5]] == alldata[[1]][jj])
		wh2 <- which(pdb_data[[6]] == alldata[[2]][jj])
		wha <- intersect(wh1, wh2)

		wh1 <- which(pdb_data[[5]] == alldata[[2]][jj])
		wh2 <- which(pdb_data[[6]] == alldata[[1]][jj])
		whb <- intersect(wh1, wh2)

		wh <- union(wha, whb)
		tempd <- unique(pdb_data[wh, c(3,4,5,6)])[1,]

		if(alldata[[1]][jj] == tempd[[3]][1]){
			uniprot1x <- c(uniprot1x, tempd[[1]][1])
			uniprot2x <- c(uniprot2x, tempd[[2]][1])
		}else{
			uniprot1x <- c(uniprot1x, tempd[[2]][1])
			uniprot2x <- c(uniprot2x, tempd[[1]][1])
		}

	}

	##--------------------------------------------

	##-- filter the positives based on selected chain pairs ---
	whall <- c()
	for(ii in 1:length(alldata[[1]])){
		wh1 <- which(pdb_data$CHAIN1 == alldata[[1]][ii])
		wh2 <- which(pdb_data$CHAIN2 == alldata[[2]][ii])
		wha <- intersect(wh1, wh2)

		wh1 <- which(pdb_data$CHAIN1 == alldata[[2]][ii])
		wh2 <- which(pdb_data$CHAIN2 == alldata[[1]][ii])
		whb <- intersect(wh1, wh2)

		whc <- union(wha, whb)


		wh1 <- which(pdb_data$UNIPROT1 == uniprot1x[ii])
		wh2 <- which(pdb_data$UNIPROT2 == uniprot2x[ii])
		wha <- intersect(wh1, wh2)

		wh1 <- which(pdb_data$UNIPROT1 == uniprot2x[ii])
		wh2 <- which(pdb_data$UNIPROT2 == uniprot1x[ii])
		whb <- intersect(wh1, wh2)

		whd <- union(wha, whb)

		whe <- intersect(whd, whc)

		whall <- union(whall, whe)
	}

	pdb_data <- pdb_data[whall, ]
	data.table::fwrite(pdb_data, paste0(store_dir,'/CONTACT_positives.txt'), sep='\t', quote=FALSE)
	##---------------------------------------------------------


	alldatax <- paste0(alldata[[1]], '_', unlist(lapply( strsplit(alldata[[2]], '_'),'[[',2)))

	##---- 80/20 split ------------------------------------------
	kfold <- 5
	rndorder <- sample(seq(1, length(alldatax))) ## shuffling the order
	alldata1 <- alldatax[rndorder]
	uniprot11 <- uniprot1x[rndorder]
	uniprot22 <- uniprot2x[rndorder]

	sets <- list()
	uni1 <- list()
	uni2 <- list()
	split <- ceiling(length(alldata1)/kfold)

	start <- 1
	for(k in 1:kfold){
		if(k != kfold){
			end <- start+split-1
			sets[[k]] <- alldata1[start:end]
			uni1[[k]] <- uniprot11[start:end]
			uni2[[k]] <- uniprot22[start:end]
			start <- end+1
		}else{
			sets[[k]] <- alldata1[start:length(alldata1)]
			uni1[[k]] <- uniprot11[start:length(alldata1)]
			uni2[[k]] <- uniprot22[start:length(alldata1)]
		}
	}


	for(k in 1:length(sets)){
		test <- sets[[k]]
		train <- c()

		for(j in 1:length(sets)){

			if(j!=k){
				train <- union(train, sets[[j]])
			}
		}

		data.table::fwrite(data.frame(test), paste0(store_dir,'/test', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE)
		data.table::fwrite(data.frame(train), paste0(store_dir,'/train', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE)
		data.table::fwrite(data.frame(uni1[[k]], uni2[[k]]), paste0(store_dir,'/test_info', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')

	}

}


###############################################################################################
# The following code is not finally tested on different machines. It is generating PISA and EPPIC data
###############################################################################################




if(FALSE){
## GT data : PISA definitions -----
gtdata <- 'PISA_EEIN_0.5'

store_dir <- paste0('../data/GT_datasets/',gtdata)
dir.create(store_dir,recursive=TRUE)

pdb_inter <- fread(paste0('../data/PISA_networks/',gtdata,'_negatives.txt'), sep='\t')
wh <- union(which(pdb_inter[[1]] == ''), which(pdb_inter[[2]] == ''))
if(length(wh) > 0){
	pdb_inter <- pdb_inter[-wh, ]
}

ch1 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN1)
ch2 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN2)
pdb_data <- data.frame(EXON1=pdb_inter$exon1, EXON2=pdb_inter$exon2, UNIPROT1=pdb_inter$protein1, UNIPROT2=pdb_inter$protein2, CHAIN1=ch1, CHAIN2=ch2)
data.table::fwrite(pdb_data, paste0(store_dir,'/PISA_negatives.txt'), sep='\t', quote=FALSE)

pdb_inter <- fread(paste0('../data/PISA_networks_filtered/',gtdata,'_positives.txt'), sep='\t')
wh <- union(which(pdb_inter[[1]] == ''), which(pdb_inter[[2]] == ''))
if(length(wh) > 0){
	pdb_inter <- pdb_inter[-wh, ]
}
ch1 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN1)
ch2 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN2)
pdb_data <- data.frame(EXON1=pdb_inter$exon1, EXON2=pdb_inter$exon2, UNIPROT1=pdb_inter$protein1, UNIPROT2=pdb_inter$protein2, CHAIN1=ch1, CHAIN2=ch2)


nee <- igraph::simplify(igraph::graph_from_data_frame(pdb_data[,c(1,2)], directed=FALSE))

pdbids1 <- unlist(lapply(strsplit(pdb_data[[5]], '[_]'), '[[', 1))
chains1 <- unlist(lapply(strsplit(pdb_data[[5]], '[_]'), '[[', 2))

pdbids2 <- unlist(lapply(strsplit(pdb_data[[6]], '[_]'), '[[', 1))
chains2 <- unlist(lapply(strsplit(pdb_data[[6]], '[_]'), '[[', 2))

## read the chains mapping from the previously processed data for contact_6_1
alldx <- data.table::fread(chain_path, header=FALSE)
store_pdbid <- alldx[[1]]
org_chain <- alldx[[2]]
new_chain <- alldx[[3]]

## ---- replace the orginal chain with the new chain -----------
for(k in 1:length(store_pdbid)){
	wh1 <- which(pdbids1 == store_pdbid[k])
	wh2 <- which(chains1 == org_chain[k])
	wh <- intersect(wh1, wh2)
	if(length(wh) != 0){
		chains1[wh] <- new_chain[k]
	}

	wh2 <- which(chains2 == org_chain[k])
	wh <- intersect(wh1, wh2)
	if(length(wh) != 0){
		chains2[wh] <- new_chain[k]
	}
}

pdbchains1 <- paste0(pdbids1,'_',chains1)
pdbchains2 <- paste0(pdbids2,'_',chains2)

##--- update pdb_data with new chains
pdb_data$CHAIN1 <- pdbchains1
pdb_data$CHAIN2 <- pdbchains2

# data.table::fwrite(pdb_data, paste0(store_dir,'/PISA_positives.txt'), sep='\t', quote=FALSE)

##---- save list of training and testing data ---------------
pisa_data <- igraph::simplify(igraph::graph_from_data_frame(data.frame(pdbchains1, pdbchains2), directed=FALSE))

##---- filter to keep protein pairs processed by all --------
pisa_filtered <- igraph::intersection(pisa_data)#, pisa_to_use)

alldata <- data.frame(igraph::as_edgelist(pisa_filtered))

# ##---------------------------------------------------------

##--- get the corresponding uniprot ids ---
uniprot1x <- c()
uniprot2x <- c()

for(jj in 1:length(alldata[[1]])){

	wh1 <- which(pdb_data[[5]] == alldata[[1]][jj])
	wh2 <- which(pdb_data[[6]] == alldata[[2]][jj])
	wha <- intersect(wh1, wh2)

	wh1 <- which(pdb_data[[5]] == alldata[[2]][jj])
	wh2 <- which(pdb_data[[6]] == alldata[[1]][jj])
	whb <- intersect(wh1, wh2)

	wh <- union(wha, whb)

	tempd <- unique(pdb_data[wh, c(3,4,5,6)])[1,]

	if(alldata[[1]][jj] == tempd[[3]][1]){
		uniprot1x <- c(uniprot1x, tempd[[1]][1])
		uniprot2x <- c(uniprot2x, tempd[[2]][1])
	}else{
		uniprot1x <- c(uniprot1x, tempd[[2]][1])
		uniprot2x <- c(uniprot2x, tempd[[1]][1])
	}


}

##--------------------------------------------

##-- filter the positives based on selected chain pairs ---
whall <- c()
for(ii in 1:length(alldata[[1]])){
	wh1 <- which(pdb_data$CHAIN1 == alldata[[1]][ii])
	wh2 <- which(pdb_data$CHAIN2 == alldata[[2]][ii])
	wha <- intersect(wh1, wh2)

	wh1 <- which(pdb_data$CHAIN1 == alldata[[2]][ii])
	wh2 <- which(pdb_data$CHAIN2 == alldata[[1]][ii])
	whb <- intersect(wh1, wh2)

	whc <- union(wha, whb)


	wh1 <- which(pdb_data$UNIPROT1 == uniprot1x[ii])
	wh2 <- which(pdb_data$UNIPROT2 == uniprot2x[ii])
	wha <- intersect(wh1, wh2)

	wh1 <- which(pdb_data$UNIPROT1 == uniprot2x[ii])
	wh2 <- which(pdb_data$UNIPROT2 == uniprot1x[ii])
	whb <- intersect(wh1, wh2)

	whd <- union(wha, whb)

	whe <- intersect(whd, whc)

	whall <- union(whall, whe)
}

pdb_data <- pdb_data[whall, ]
data.table::fwrite(pdb_data, paste0(store_dir,'/PISA_positives.txt'), sep='\t', quote=FALSE)
##---------------------------------------------------------


alldatax <- paste0(alldata[[1]], '_', unlist(lapply( strsplit(alldata[[2]], '_'),'[[',2)))

##---- 80/20 split ------------------------------------------
kfold <- 5
rndorder <- sample(seq(1, length(alldatax))) ## shuffling the order
alldata1 <- alldatax[rndorder]
uniprot11 <- uniprot1x[rndorder]
uniprot22 <- uniprot2x[rndorder]

sets <- list()
uni1 <- list()
uni2 <- list()
split <- ceiling(length(alldata1)/kfold)

start <- 1
for(k in 1:kfold){
	if(k != kfold){
		end <- start+split-1
		sets[[k]] <- alldata1[start:end]
		uni1[[k]] <- uniprot11[start:end]
		uni2[[k]] <- uniprot22[start:end]
		start <- end+1
	}else{
		sets[[k]] <- alldata1[start:length(alldata1)]
		uni1[[k]] <- uniprot11[start:length(alldata1)]
		uni2[[k]] <- uniprot22[start:length(alldata1)]
	}
}


for(k in 1:length(sets)){
	test <- sets[[k]]
	train <- c()

	for(j in 1:length(sets)){

		if(j!=k){
			train <- union(train, sets[[j]])
		}
	}

	data.table::fwrite(data.frame(test), paste0(store_dir,'/test', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE)
	data.table::fwrite(data.frame(train), paste0(store_dir,'/train', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE)
	data.table::fwrite(data.frame(uni1[[k]], uni2[[k]]), paste0(store_dir,'/test_info', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')

}






## GT data : EPPIC definitions ------------------------------------

store_dir <- paste0('../data/GT_datasets/EPPIC_EEIN_filtered')
dir.create(store_dir,recursive=TRUE)

pdb_inter <- data.table::fread(paste0('../data/EPPIC_EEIN_negative.txt'), sep='\t')
wh <- union(which(pdb_inter[[1]] == ''), which(pdb_inter[[2]] == ''))
if(length(wh) > 0){
	pdb_inter <- pdb_inter[-wh, ]
}
ch1 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN1)
ch2 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN2)
pdb_data <- data.frame(EXON1=pdb_inter$exon1, EXON2=pdb_inter$exon2, UNIPROT1=pdb_inter$protein1, UNIPROT2=pdb_inter$protein2, CHAIN1=ch1, CHAIN2=ch2)
data.table::fwrite(pdb_data, paste0(store_dir,'/EPPIC_negatives.txt'), sep='\t', quote=FALSE)

pdb_inter <- data.table::fread(paste0('../data/EPPIC_EEIN_filtered_positive.txt'), sep='\t')
wh <- union(which(pdb_inter[[1]] == ''), which(pdb_inter[[2]] == ''))
if(length(wh) > 0){
	pdb_inter <- pdb_inter[-wh, ]
}
ch1 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN1)
ch2 <- paste0(pdb_inter$PDBID,'_',pdb_inter$CHAIN2)
pdb_data <- data.frame(EXON1=pdb_inter$exon1, EXON2=pdb_inter$exon2, UNIPROT1=pdb_inter$protein1, UNIPROT2=pdb_inter$protein2, CHAIN1=ch1, CHAIN2=ch2)


nee <- igraph::simplify(igraph::graph_from_data_frame(pdb_data[,c(1,2)], directed=FALSE))

pdbids1 <- unlist(lapply(strsplit(pdb_data[[5]], '[_]'), '[[', 1))
chains1 <- unlist(lapply(strsplit(pdb_data[[5]], '[_]'), '[[', 2))

pdbids2 <- unlist(lapply(strsplit(pdb_data[[6]], '[_]'), '[[', 1))
chains2 <- unlist(lapply(strsplit(pdb_data[[6]], '[_]'), '[[', 2))

## read the chains mapping from the previously processed data for contact_6_1
alldx <- data.table::fread(chain_path, header=FALSE)
store_pdbid <- alldx[[1]]
org_chain <- alldx[[2]]
new_chain <- alldx[[3]]

## ---- replace the orginal chain with the new chain -----------
for(k in 1:length(store_pdbid)){
	wh1 <- which(pdbids1 == store_pdbid[k])
	wh2 <- which(chains1 == org_chain[k])
	wh <- intersect(wh1, wh2)
	if(length(wh) != 0){
		chains1[wh] <- new_chain[k]
	}

	wh2 <- which(chains2 == org_chain[k])
	wh <- intersect(wh1, wh2)
	if(length(wh) != 0){
		chains2[wh] <- new_chain[k]
	}
}

pdbchains1 <- paste0(pdbids1,'_',chains1)
pdbchains2 <- paste0(pdbids2,'_',chains2)

##--- update pdb_data with new chains
pdb_data$CHAIN1 <- pdbchains1
pdb_data$CHAIN2 <- pdbchains2

# data.table::fwrite(pdb_data, paste0(store_dir,'/EPPIC_positives.txt'), sep='\t', quote=FALSE)

##---- save list of training and testing data ---------------
eppic_data <- igraph::simplify(igraph::graph_from_data_frame(data.frame(pdbchains1, pdbchains2), directed=FALSE))

##---- filter to keep protein pairs processed by all --------
eppic_filtered <- igraph::intersection(eppic_data)#, eppic_to_use)

alldata <- data.frame(igraph::as_edgelist(eppic_filtered))

# ##---------------------------------------------------------

##--- get the corresponding uniprot ids ---
uniprot1x <- c()
uniprot2x <- c()

for(jj in 1:length(alldata[[1]])){

	wh1 <- which(pdb_data[[5]] == alldata[[1]][jj])
	wh2 <- which(pdb_data[[6]] == alldata[[2]][jj])
	wha <- intersect(wh1, wh2)

	wh1 <- which(pdb_data[[5]] == alldata[[2]][jj])
	wh2 <- which(pdb_data[[6]] == alldata[[1]][jj])
	whb <- intersect(wh1, wh2)

	wh <- union(wha, whb)

	tempd <- unique(pdb_data[wh, c(3,4,5,6)])[1,]

	if(alldata[[1]][jj] == tempd[[3]][1]){
		uniprot1x <- c(uniprot1x, tempd[[1]][1])
		uniprot2x <- c(uniprot2x, tempd[[2]][1])
	}else{
		uniprot1x <- c(uniprot1x, tempd[[2]][1])
		uniprot2x <- c(uniprot2x, tempd[[1]][1])
	}

}

##--------------------------------------------

##-- filter the positives based on selected chain pairs ---
whall <- c()
for(ii in 1:length(alldata[[1]])){
	wh1 <- which(pdb_data$CHAIN1 == alldata[[1]][ii])
	wh2 <- which(pdb_data$CHAIN2 == alldata[[2]][ii])
	wha <- intersect(wh1, wh2)

	wh1 <- which(pdb_data$CHAIN1 == alldata[[2]][ii])
	wh2 <- which(pdb_data$CHAIN2 == alldata[[1]][ii])
	whb <- intersect(wh1, wh2)

	whc <- union(wha, whb)


	wh1 <- which(pdb_data$UNIPROT1 == uniprot1x[ii])
	wh2 <- which(pdb_data$UNIPROT2 == uniprot2x[ii])
	wha <- intersect(wh1, wh2)

	wh1 <- which(pdb_data$UNIPROT1 == uniprot2x[ii])
	wh2 <- which(pdb_data$UNIPROT2 == uniprot1x[ii])
	whb <- intersect(wh1, wh2)

	whd <- union(wha, whb)

	whe <- intersect(whd, whc)

	whall <- union(whall, whe)
}

pdb_data <- pdb_data[whall, ]
data.table::fwrite(pdb_data, paste0(store_dir,'/EPPIC_positives.txt'), sep='\t', quote=FALSE)
##---------------------------------------------------------


alldatax <- paste0(alldata[[1]], '_', unlist(lapply( strsplit(alldata[[2]], '_'),'[[',2)))

##---- 80/20 split ------------------------------------------
kfold <- 5
rndorder <- sample(seq(1, length(alldatax))) ## shuffling the order
alldata1 <- alldatax[rndorder]
uniprot11 <- uniprot1x[rndorder]
uniprot22 <- uniprot2x[rndorder]

sets <- list()
uni1 <- list()
uni2 <- list()
split <- floor(length(alldata1)/kfold)

start <- 1
for(k in 1:kfold){
	if(k != kfold){
		end <- start+split-1
		sets[[k]] <- alldata1[start:end]
		uni1[[k]] <- uniprot11[start:end]
		uni2[[k]] <- uniprot22[start:end]
		start <- end+1
	}else{
		sets[[k]] <- alldata1[start:length(alldata1)]
		uni1[[k]] <- uniprot11[start:length(alldata1)]
		uni2[[k]] <- uniprot22[start:length(alldata1)]
	}
}


for(k in 1:length(sets)){
	test <- sets[[k]]
	train <- c()

	for(j in 1:length(sets)){

		if(j!=k){
			train <- union(train, sets[[j]])
		}
	}

	data.table::fwrite(data.frame(test), paste0(store_dir,'/test', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE)
	data.table::fwrite(data.frame(train), paste0(store_dir,'/train', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE)
	data.table::fwrite(data.frame(uni1[[k]], uni2[[k]]), paste0(store_dir,'/test_info', k, '.txt'), row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\t')

}




###############################################################################################
# Use the following code to exclude protein pairs which cannot be processed, has to be moved 
# to the begin of the file
###############################################################################################
##-- contact ----
allfiles <- list.files('../../data/Working_proteins', pattern='^CONTACT', full.names=TRUE)

temp1 <- data.table::fread(allfiles[1], header=FALSE)
temp2 <- data.table::fread(allfiles[2], header=FALSE)
temp3 <- data.table::fread(allfiles[3], header=FALSE)

s1 <- strsplit(temp1[[1]],'[_]')
s2 <- strsplit(temp2[[1]],'[_]')
s3 <- strsplit(temp3[[1]],'[_]')

g1 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s1, '[[', 1)),'_',unlist(lapply(s1, '[[', 2))),
	paste0(unlist(lapply(s1, '[[', 1)),'_',unlist(lapply(s1, '[[', 3)))), directed=FALSE))

g2 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s2, '[[', 1)),'_',unlist(lapply(s2, '[[', 2))),
	paste0(unlist(lapply(s2, '[[', 1)),'_',unlist(lapply(s2, '[[', 3)))), directed=FALSE))

g3 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s3, '[[', 1)),'_',unlist(lapply(s3, '[[', 2))),
	paste0(unlist(lapply(s3, '[[', 1)),'_',unlist(lapply(s3, '[[', 3)))), directed=FALSE))

g4 <- igraph::intersection(g1, igraph::intersection(g2, g3))
g5 <- igraph::simplify(g4)
contact_to_use <- g5 #data.frame(igraph::as_edgelist(g5))
# add contact_to_use by uncommenting in row with: contact_filtered <- igraph::intersection(contact_data)#, contact_to_use)

##----------------------------------------------------------------------------------------------------------------------------------------

##-- eppic ----
allfiles <- list.files('../../data/Working_proteins', pattern='^EPPIC', full.names=TRUE)

temp1 <- data.table::fread(allfiles[1], header=FALSE)
temp2 <- data.table::fread(allfiles[2], header=FALSE)
temp3 <- data.table::fread(allfiles[3], header=FALSE)

s1 <- strsplit(temp1[[1]],'[_]')
s2 <- strsplit(temp2[[1]],'[_]')
s3 <- strsplit(temp3[[1]],'[_]')

g1 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s1, '[[', 1)),'_',unlist(lapply(s1, '[[', 2))),
	paste0(unlist(lapply(s1, '[[', 1)),'_',unlist(lapply(s1, '[[', 3)))), directed=FALSE))

g2 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s2, '[[', 1)),'_',unlist(lapply(s2, '[[', 2))),
	paste0(unlist(lapply(s2, '[[', 1)),'_',unlist(lapply(s2, '[[', 3)))), directed=FALSE))

g3 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s3, '[[', 1)),'_',unlist(lapply(s3, '[[', 2))),
	paste0(unlist(lapply(s3, '[[', 1)),'_',unlist(lapply(s3, '[[', 3)))), directed=FALSE))

g4 <- igraph::intersection(g1, igraph::intersection(g2, g3))
g5 <- igraph::simplify(g4)
eppic_to_use <- g5 #data.frame(igraph::as_edgelist(g5))
# add eppic_to_use by uncommenting in row with: eppic_filtered <- igraph::intersection(eppic_data)#, eppic_to_use)

##----------------------------------------------

##-- pisa ----
allfiles <- list.files('../../data/Working_proteins', pattern='^PISA', full.names=TRUE)

temp1 <- data.table::fread(allfiles[1], header=FALSE)
temp2 <- data.table::fread(allfiles[2], header=FALSE)
temp3 <- data.table::fread(allfiles[3], header=FALSE)

s1 <- strsplit(temp1[[1]],'[_]')
s2 <- strsplit(temp2[[1]],'[_]')
s3 <- strsplit(temp3[[1]],'[_]')

g1 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s1, '[[', 1)),'_',unlist(lapply(s1, '[[', 2))),
	paste0(unlist(lapply(s1, '[[', 1)),'_',unlist(lapply(s1, '[[', 3)))), directed=FALSE))

g2 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s2, '[[', 1)),'_',unlist(lapply(s2, '[[', 2))),
	paste0(unlist(lapply(s2, '[[', 1)),'_',unlist(lapply(s2, '[[', 3)))), directed=FALSE))

g3 <- igraph::simplify(igraph::graph_from_data_frame(data.frame(paste0(unlist(lapply(s3, '[[', 1)),'_',unlist(lapply(s3, '[[', 2))),
	paste0(unlist(lapply(s3, '[[', 1)),'_',unlist(lapply(s3, '[[', 3)))), directed=FALSE))

g4 <- igraph::intersection(g1, igraph::intersection(g2, g3))
g5 <- igraph::simplify(g4)
pisa_to_use <- g5 #data.frame(igraph::as_edgelist(g5))
# add pisa_to_use by uncommenting in row with: pisa_filtered <- igraph::intersection(pisa_data)#, pisa_to_use)

##----------------------------------------------
}