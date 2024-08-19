##############################################################################################
# Purpose: save uniprot entries with other info: geneid, gene name, pdb chains, 
# resolution of chains, start of chain, end of chain
# if no resolution is present, then choose any
## A case example of wrong entry in uniprot maps
##############################################################################################

rm(list=ls())
library(seqinr)
library(data.table)
library(Rcpp)
library(stringr)
library(plyr)
library(igraph)
# turning warnings into errors
options(warn=0)

#############################################################################################################
# change datapaths and/or species here
data_abbr <- "HS" 
store_dir <- '../'
#############################################################################################################
uniprot_HS_dir <- paste0(store_dir, "uniprot_sprot_", data_abbr) # if changed, change in all scripts
allfiles <- list.files(uniprot_HS_dir, full.names=TRUE)
output_file <- paste0(store_dir, "uniprot_sequences_", data_abbr, ".txt") # if changed, change in all scripts
if(file.exists(output_file)){file.remove(output_file)}
process_file <- paste0(store_dir, "processed_complex_final.txt") # if changed, change in all scripts

### saving uniprot AA sequences
# save all sequences in a vector
uniseq_aa <- list()
for(k in 1:length(allfiles)){

	temp <- readLines(allfiles[k])
	tempseq <- substr(temp, 1,2)

	wha <- which(tempseq == 'AC')
	# take the canonical uniprot identifier
	temp1 <- unlist(lapply(strsplit(temp[wha[1]], '[;]'), '[[',1))
	temp11 <- lapply(strsplit(temp1,'\\s+'), '[[',2)
	temp_uniprot <- temp11[[1]]

	wh1 <- which(tempseq == 'SQ')+1
	wh2 <- which(tempseq == '//')-1

	temps1 <- gsub(' ','',paste(temp[wh1:wh2], collapse=''))
	write.fasta(temps1, temp_uniprot, output_file, open='a', nbchar=60, as.string=FALSE)
	cat(k,' of ', length(allfiles), ' done\n')

}


###########################################################################################################
# create a mapping of gene name, uniprot ids, geneid, pdb ids, start, and end of the uniprot sequence

cppFunction("List unimappdb1(CharacterVector pdbc, CharacterVector tempuni, NumericVector resolution, NumericVector start, NumericVector end, CharacterVector exp){

	int loop1 = pdbc.size();
	int loop2 = tempuni.size();
	int fsize = loop1*loop2;
	CharacterVector uniprotid_filt(fsize);
	CharacterVector pdbc_filt(fsize);
	NumericVector resolution_filt(fsize);
	NumericVector start_filt(fsize);
	NumericVector end_filt(fsize);
	CharacterVector exp_filt(fsize);

	int counter = 0;

	for(int k=0; k<loop1; k++){

		for(int j=0; j<loop2; j++){

			uniprotid_filt[counter] = tempuni[j];
			resolution_filt[counter] = resolution[k];
			pdbc_filt[counter] = pdbc[k];
			start_filt[counter] = start[k];
			end_filt[counter] = end[k];
			exp_filt[counter] = exp[k];
			counter = counter+1;

		}

	}

	List L = List::create(uniprotid_filt,resolution_filt, pdbc_filt, start_filt, end_filt,exp_filt);
	return L;
  
}")

cppFunction("List unimappdb2(CharacterVector pdbc, CharacterVector tempuni, NumericVector start, NumericVector end, CharacterVector exp){

	int loop1 = pdbc.size();
	int loop2 = tempuni.size();
	int fsize = loop1*loop2;
	CharacterVector uniprotid_filt(fsize);
	CharacterVector pdbc_filt(fsize);
	NumericVector start_filt(fsize);
	NumericVector end_filt(fsize);
	CharacterVector exp_filt(fsize);

	int counter = 0;

	for(int k=0; k<loop1; k++){

		for(int j=0; j<loop2; j++){

			uniprotid_filt[counter] = tempuni[j];
			pdbc_filt[counter] = pdbc[k];
			start_filt[counter] = start[k];
			end_filt[counter] = end[k];
			exp_filt[counter] = exp[k];
			counter = counter+1;

		}

	}

	List L = List::create(uniprotid_filt, pdbc_filt, start_filt, end_filt,exp_filt);
	return L;
  
}")


rproc1 <- function(temp2, resolution, pdbid, temp_uniprot){

	temp3 <- strsplit(trimws(unlist(lapply(temp2, '[[', 5))),'[=]')
	tempe <- trimws(unlist(lapply(temp2, '[[', 3)))
	## choose temp3 entries with at least two length because that means there is chain information
	temp3_count <- lengths(temp3)
	temp3_wh <- which(temp3_count == 2) # only taking the proteins with continuous information

	if(length(temp3_wh) > 0){

		temp3 <- temp3[temp3_wh]
		tempe <- tempe[temp3_wh]
		resolution <- resolution[temp3_wh]
		pdbid <- pdbid[temp3_wh]
		temp4 <- unlist(lapply(temp3, '[[', 1))
		chain <- unlist(lapply(strsplit(temp4, '[/]'),'[[',1))
		temp5 <- unlist(lapply(temp3, '[[', 2))
		temp6 <- paste0(unlist(lapply(strsplit(temp5, '[.]'),'[[',1)),'-') #
		start <- unlist(lapply(strsplit(temp6, '[-]'),'[[',1))
		end <- unlist(lapply(strsplit(temp6, '[-]'),'[[',2))
		# replacing '' value in start and end vector
		start[start == ''] <- 0 
		end[end == ''] <- 0
		wh10 <- which(resolution <= 3)

		pdbid <- pdbid[wh10]
		chain <- chain[wh10]
		resolution <- resolution[wh10]
		start <- start[wh10]
		end <- end[wh10]
		tempe <- tempe[wh10]
		pdbc <- paste0(tolower(pdbid), '_', chain)
		## all uniprots
		tempuni <- temp_uniprot
		
		tempxx <- unimappdb1(pdbc, tempuni, as.numeric(resolution), as.numeric(start), as.numeric(end), tempe)

		return(tempxx)

	}else{
		return(NULL)
	}

}

rproc2 <- function(temp2, pdbid, temp_uniprot){

	temp3 <- strsplit(trimws(unlist(lapply(temp2, '[[', 5))),'[=]')
	tempe <- trimws(unlist(lapply(temp2, '[[', 3)))

	# choose temp3 entries with at least two length because that means there is chain information
	temp3_count <- lengths(temp3)
	temp3_wh <- which(temp3_count == 2) # only taking the proteins with continuous information

	if(length(temp3_wh) > 0){

		temp3 <- temp3[temp3_wh]
		tempe <- tempe[temp3_wh]
		pdbid <- pdbid[temp3_wh]
		temp4 <- unlist(lapply(temp3, '[[', 1))
		chain <- unlist(lapply(strsplit(temp4, '[/]'),'[[',1))
		temp5 <- unlist(lapply(temp3, '[[', 2))
		temp6 <- paste0(unlist(lapply(strsplit(temp5, '[.]'),'[[',1)),'-') #
		start <- unlist(lapply(strsplit(temp6, '[-]'),'[[',1))
		end <- unlist(lapply(strsplit(temp6, '[-]'),'[[',2))

		# replacing '' value in start and end vector
		start[start == ''] <- 0 
		end[end == ''] <- 0
		
		pdbc <- paste0(tolower(pdbid), '_', chain)
		## all uniprots
		tempuni <- temp_uniprot
		
		tempxx <- unimappdb2(pdbc, tempuni, as.numeric(start), as.numeric(end), tempe)

		return(tempxx)

	}else{
		return(NULL)
	}
	
}

uniprotid1 <- c()
genename1 <- c()
pdbchain1 <- c()
resol1 <- c()
allstart1 <- c()
allend1 <- c()
expr1 <- c()

uniprotid2 <- c()
genename2 <- c()
pdbchain2 <- c()
allstart2 <- c()
allend2 <- c()
expr2 <- c()

pdb_entry <- 0
pdb_str <- 0
pdb_xr <- 0
pdb_nmr <- 0
expr_info <- c()
pdb_files <- c()

for(k in 1:length(allfiles)){

	temp <- readLines(allfiles[k])
	tempcc <- substr(temp, 1,2)
	
	# all uniprot ids
	wha <- which(tempcc == 'AC')
	# take the canonical uniprot identifier
	temp1 <- unlist(lapply(strsplit(temp[wha[1]], '[;]'), '[[',1))
	temp11 <- lapply(strsplit(temp1,'\\s+'), '[[',2)
	temp_uniprot <- temp11[[1]]

	# all pdb
	whp <- which(temp %like% ' PDB;')

	if(length(whp) == 0){
		next # skip if no pdb antry is present
	}else{
		pdb_entry <- pdb_entry+1
		temp_pdb <- temp[whp]
		temp2 <- strsplit(temp_pdb,'[;]')
		experiment <- trimws(unlist(lapply(temp2, '[[', 3)))
		expr_info <- union(expr_info, experiment)

		## only keeping entries with X-ray, EM or NMR
		whx <- which(experiment %in% c('X-ray', 'EM', 'NMR'))
		if(length(whx) != 0){ pdb_str <- pdb_str+1 }
		temp_pdb <- temp_pdb[whx]
		temp2 <- strsplit(temp_pdb,'[;]')
		pdbid <- trimws(unlist(lapply(temp2, '[[', 2)))
		pdb_files <- union(pdb_files,unique(pdbid))
		resolution <- substr(trimws(unlist(lapply(temp2, '[[', 4))),1,4)
		resolution[resolution == '-'] <- 100
		resolution <- as.numeric(resolution)
		whr <- which(resolution <= 3)

		if(length(whr) != 0){

			pdb_xr <- pdb_xr+1

			temp_pdbx <- temp_pdb[whr]
			temp2 <- strsplit(temp_pdbx,'[;]')

			tempxx <- rproc1(temp2, resolution[whr], pdbid[whr], temp_uniprot)
			if(is.null(tempxx)){next}
			uniprotid1 <- c(uniprotid1, tempxx[[1]])
			pdbchain1 <- c(pdbchain1, tempxx[[3]])
			allstart1 <- c(allstart1, tempxx[[4]])
			allend1 <- c(allend1, tempxx[[5]])
			resol1 <- c(resol1, tempxx[[2]])
			expr1 <- c(expr1, tempxx[[6]])

		}else{
			## check if the resolution is 100
			whr <- which(resolution == 100)
			if(length(whr) != 0){

				pdb_nmr <- pdb_nmr+1

				temp_pdbx <- temp_pdb[whr]
				temp2 <- strsplit(temp_pdbx,'[;]')
	
				tempxx <- rproc2(temp2, pdbid[whr], temp_uniprot)
				if(is.null(tempxx)){next}
				uniprotid2 <- c(uniprotid2, tempxx[[1]])
				pdbchain2 <- c(pdbchain2, tempxx[[2]])
				allstart2 <- c(allstart2, tempxx[[3]])
				allend2 <- c(allend2, tempxx[[4]])
				expr2 <- c(expr2, tempxx[[5]])

			}
		}

	}
	cat(k,' of ', length(allfiles), ' done\n')
}

id_map1 <- data.frame(uniprotkbac=uniprotid1, pdbchain=pdbchain1, resolution=resol1, start=allstart1, end=allend1, exp=expr1)
id_map2 <- data.frame(uniprotkbac=uniprotid2, pdbchain=pdbchain2, resolution=rep(0,length(uniprotid2)), 
	start=allstart2, end=allend2, exp=expr2)


id_map1$PDBID <- unlist(lapply(strsplit(id_map1$pdbchain, '[_]'), '[[', 1))
id_map2$PDBID <- unlist(lapply(strsplit(id_map2$pdbchain, '[_]'), '[[', 1))
id_map1$CHAIN <- unlist(lapply(strsplit(id_map1$pdbchain, '[_]'), '[[', 2))
id_map2$CHAIN <- unlist(lapply(strsplit(id_map2$pdbchain, '[_]'), '[[', 2))
id_map1$ID <- paste0(id_map1$PDBID)#,'_',id_map1$exp,'_',id_map1$resolution)
id_map2$ID <- paste0(id_map2$PDBID)#,'_',id_map2$exp,'_',id_map2$resolution)


##-- all pdb ids
pdbids1 <- unique(id_map1[[7]])
pdbids2 <- unique(id_map2[[7]])
allpdbids <- union(pdbids1, pdbids2)


## -- choose the PDBIDs that results in the highest number of complexes ---

allpdbs <- plyr::count(id_map1$ID)
allpdbsu <- allpdbs[allpdbs$freq > 1, ]

p1 <- c()
p2 <- c()
pdbid <- c()
chainid1 <- c()
chainid2 <- c()
resol <- c()
st1 <- c()
st2 <- c()
ed1 <- c()
ed2 <- c()

##-- for four of the PDB IDs, the PDB resolution info is not consistant (marginal differences in the values)
idx <- c()
for(k in 1:length(allpdbsu[[1]])){

	temp <- id_map1[id_map1$ID == allpdbsu[[1]][k], ]
	# if(length(temp[[1]]) > 1){break}
	loop1 <- length(temp[[1]])-1
	loop2 <- length(temp[[1]])
	loop3 <- (loop1*loop2)/2

	for(i in 1:loop1){
		t1 <- temp[i,]
		m <- i+1
		for(j in m:loop2){
			t2 <- temp[j,]
			p1 <- c(p1, t1$uniprotkbac)
			chainid1 <- c(chainid1, t1$CHAIN)
			st1 <- c(st1, t1$start)
			ed1 <- c(ed1, t1$end)
			p2 <- c(p2, t2$uniprotkbac)
			chainid2 <- c(chainid2, t2$CHAIN)
			st2 <- c(st2, t2$start)
			ed2 <- c(ed2, t2$end)
		}
	}

	pdbid <- c(pdbid, rep(unique(temp$ID), loop3))
	# if(length(unique(temp$resolution)) > 1){idx <- c(idx,k)}
	resol <- c(resol, rep(unique(temp$resolution)[1], loop3))

	cat(k,' of ', length(allpdbsu[[1]]), ' done\n')

}

cmx_data1 <- data.frame(ID=pdbid, RESOL=resol, PROTEIN1=p1, CHAIN1=chainid1, START1=st1, END1=ed1,
	PROTEIN2=p2, CHAIN2=chainid2, START2=st2, END2=ed2)

##--unique complexes
cmx1 <- igraph::simplify(igraph::graph_from_data_frame(cmx_data1[,c(3,7)], directed=FALSE))
cmx11 <- igraph::as_data_frame(cmx1)


##---choose a pdb id for each complex -----
allData <- data.frame(matrix(ncol=length(cmx_data1), nrow=0))
for(k in 1:length(cmx11[[1]])){
	x <- cmx11[[1]][k]
	y <- cmx11[[2]][k]
	wh1 <- which(cmx_data1$PROTEIN1 == x)
	wh2 <- which(cmx_data1$PROTEIN2 == y)
	wha <- intersect(wh1, wh2)
	wh1 <- which(cmx_data1$PROTEIN2 == x)
	wh2 <- which(cmx_data1$PROTEIN1 == y)
	whb <- intersect(wh1, wh2)
	wh <- union(wha, whb)
	temp1 <- cmx_data1[wh,]

	# if multiple
	if(nrow(temp1) > 1){
		cov1 <- c()
		cov2 <- c()
		for(j in 1:nrow(temp1)){
			temp2 <- temp1[j,]
			if(x == temp2$PROTEIN1){
				cov1 <- c(cov1, abs(as.numeric(temp2$START1) - as.numeric(temp2$END1)))
				cov2 <- c(cov2, abs(as.numeric(temp2$START2) - as.numeric(temp2$END2)))
			}else{
				cov1 <- c(cov1, abs(as.numeric(temp2$START2) - as.numeric(temp2$END2)))
				cov2 <- c(cov2, abs(as.numeric(temp2$START1) - as.numeric(temp2$END1)))
			}
		}

		# max coverage
		cov1_mx <- which(cov1 == max(cov1))
		cov2_mx <- which(cov2 == max(cov2))
		cov_f <- intersect(cov1_mx, cov2_mx)

		# if no intersection
		if(length(cov_f) == 0){
			if(cov1[cov1_mx[1]] < cov2[cov2_mx[1]]){ # choose the position that has higher coverage for the smaller protein
				# because this ensures that we choose loss of info in longer protein than the shorter proteins, 
				# potetially leading to less percentage-loss of a given protein
				cov_f <- cov1_mx[1]
			}else{
				cov_f <- cov2_mx[1]
			}
		}
		temp3 <- temp1[cov_f, ]

		# if now more than one row, then select by pdb resolution
		whr <- which(temp3$RESOL == min(temp3$RESOL))
		temp4 <- temp3[whr[1], ]
		allData <- rbind(allData, temp4)
	}else{
		allData <- rbind(allData, temp1)
	}
	cat(k,' of ', length(cmx11[[1]]), ' done\n')
}

netpdb1 <- unique(allData[[1]])
## -- choose the PDBIDs that results in the highest number of complexes ---
allpdbs <- plyr::count(id_map2$ID)
allpdbsu <- allpdbs[allpdbs$freq > 1, ]

p1 <- c()
p2 <- c()
pdbid <- c()
chainid1 <- c()
chainid2 <- c()
resol <- c()
st1 <- c()
st2 <- c()
ed1 <- c()
ed2 <- c()

for(k in 1:length(allpdbsu[[1]])){

	temp <- id_map2[id_map2$ID == allpdbsu[[1]][k], ]
	# if(length(temp[[1]]) > 1){break}
	loop1 <- length(temp[[1]])-1
	loop2 <- length(temp[[1]])
	loop3 <- (loop1*loop2)/2

	for(i in 1:loop1){
		t1 <- temp[i,]
		m <- i+1
		for(j in m:loop2){
			t2 <- temp[j,]
			p1 <- c(p1, t1$uniprotkbac)
			chainid1 <- c(chainid1, t1$CHAIN)
			st1 <- c(st1, t1$start)
			ed1 <- c(ed1, t1$end)
			p2 <- c(p2, t2$uniprotkbac)
			chainid2 <- c(chainid2, t2$CHAIN)
			st2 <- c(st2, t2$start)
			ed2 <- c(ed2, t2$end)
		}
	}

	pdbid <- c(pdbid, rep(unique(temp$ID), loop3))
	resol <- c(resol, rep(unique(temp$resolution), loop3))

	cat(k,' of ', length(allpdbsu[[1]]), ' done\n')

}

cmx_data2 <- data.frame(ID=pdbid, RESOL=resol, PROTEIN1=p1, CHAIN1=chainid1, START1=st1, END1=ed1,
	PROTEIN2=p2, CHAIN2=chainid2, START2=st2, END2=ed2)
##--unique complexes
cmx2 <- igraph::simplify(igraph::graph_from_data_frame(cmx_data2[,c(3,7)], directed=FALSE))
cmx11 <- igraph::as_data_frame(cmx2)


##---choose a pdb id for each complex -----
allData2 <- data.frame(matrix(ncol=length(cmx_data2), nrow=0))
for(k in 1:length(cmx11[[1]])){
	x <- cmx11[[1]][k]
	y <- cmx11[[2]][k]
	wh1 <- which(cmx_data2$PROTEIN1 == x)
	wh2 <- which(cmx_data2$PROTEIN2 == y)
	wha <- intersect(wh1, wh2)
	wh1 <- which(cmx_data2$PROTEIN2 == x)
	wh2 <- which(cmx_data2$PROTEIN1 == y)
	whb <- intersect(wh1, wh2)
	wh <- union(wha, whb)
	temp1 <- cmx_data2[wh,]

	# if multiple
	if(nrow(temp1) > 1){
		cov1 <- c()
		cov2 <- c()
		for(j in 1:nrow(temp1)){
			temp2 <- temp1[j,]
			if(x == temp2$PROTEIN1){
				cov1 <- c(cov1, abs(as.numeric(temp2$START1) - as.numeric(temp2$END1)))
				cov2 <- c(cov2, abs(as.numeric(temp2$START2) - as.numeric(temp2$END2)))
			}else{
				cov1 <- c(cov1, abs(as.numeric(temp2$START2) - as.numeric(temp2$END2)))
				cov2 <- c(cov2, abs(as.numeric(temp2$START1) - as.numeric(temp2$END1)))
			}
		}

		# max coverage
		cov1_mx <- which(cov1 == max(cov1))
		cov2_mx <- which(cov2 == max(cov2))
		cov_f <- intersect(cov1_mx, cov2_mx)

		# if no intersection
		if(length(cov_f) == 0){
			if(cov1[cov1_mx[1]] < cov2[cov2_mx[1]]){ # choose the position that has higher coverage for the smaller protein
				# because this says that we choose loss of info in longer protein than the shorter proteins, 
				# potetially leading to less percentage-loss of a given protein
				cov_f <- cov1_mx[1]
			}else{
				cov_f <- cov2_mx[1]
			}
		}
		temp3 <- temp1[cov_f, ]
		# if now more than one row, then select by pdb resolution
		whr <- which(temp3$RESOL == min(temp3$RESOL))
		temp4 <- temp3[whr[1], ]
		allData2 <- rbind(allData2, temp4)
	}else{
		allData2 <- rbind(allData2, temp1)
	}
	cat(k,' of ', length(cmx11[[1]]), ' done\n')
}

netpdb2 <- unique(allData2[[1]])
netpdb <- union(netpdb1, netpdb2)

finaldata <- rbind(allData2, allData)
# cmx2 <- igraph::simplify(igraph::graph_from_data_frame(finaldata[,c(3,7)], directed=FALSE))
# cmx11 <- igraph::as_data_frame(cmx2)

fwrite(finaldata, process_file, sep='\t', row.names=FALSE, quote=FALSE)


