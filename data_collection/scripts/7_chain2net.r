##############################################################################################
# Purpose: PDB chains to networks
##############################################################################################

rm(list=ls())
library(data.table)
library(stringr)
library(plyr)
library(Rcpp)
library(ggplot2)
library(igraph)
# turning warnings into errors
options(warn=2)
########################################################################################
# change datapaths here
data_dir <- "../"
ensembl_out <- paste0(data_dir, "uniprot_pdb_Ensembl_final.txt")
store_dir <- paste0(data_dir, "uniprot_EnsemblExonPDB_map")

pdbDirectory <- paste0(data_dir, 'PDB_CIF')

# set cutoff in A between two interacting amino acids
# cutoff <- c(4, 5, 6, 7, 8)
cutoff <- c(6)

########################################################################################
## PDB to unweighted network function
cppFunction("List pdb2net(CharacterVector uniqueids, NumericVector lastPositions, NumericVector xcoord, NumericVector ycoord, NumericVector zcoord, int cutoff){

	int loop1 = uniqueids.size()-1;
	int loop2 = uniqueids.size();
	int esize = loop2*loop2;
	CharacterVector p1(esize);
	CharacterVector p2(esize);
	NumericVector dis(esize);
	int start = 0;
	int counter = 0;

	for(int k=0; k<loop1; k++){

		int i = k+1;

		for(int j=i; j<loop2; j++){

			int startc = lastPositions[j-1]+1;
			int endc = lastPositions[j];
			int startr = start;
			int endr = lastPositions[k];
			double mindist = 100;

			for(int x=startr; x<=endr; x++){

				double xx = xcoord[x];
				double xy = ycoord[x];
				double xz = zcoord[x];

				for(int y=startc; y<=endc; y++){

					double yx = xcoord[y];
					double yy = ycoord[y];
					double yz = zcoord[y];

					double adist = sqrt(pow((yx-xx),2)+pow((yy-xy),2)+pow((yz-xz),2));
					if(adist < mindist){
						mindist = adist;
					}

				}
			}

			if(mindist <= cutoff){
				p1[counter] = uniqueids[k];
				p2[counter] = uniqueids[j];
				dis[counter] = mindist;
				counter = counter+1;
			}

		}
		start = lastPositions[k]+1;

	}

	List L = List::create(p1,p2,dis,counter);
	return L;
  
}")

## all mapped data
allfiles <- list.files(store_dir, full.names=TRUE)
afile <- fread(ensembl_out, header=TRUE)
afile1 <- data.frame(chain1=paste0(afile$ID,'_',afile$CHAIN1),chain2=paste0(afile$ID,'_',afile$CHAIN2))
ig <- graph_from_data_frame(afile1, directed = FALSE)
igg <- simplify(ig, remove.loops=FALSE, remove.multiple=TRUE)
afile2 <- as.data.frame(get.edgelist(igg))
afile2$pdbid <- tolower(unlist(lapply(strsplit(afile2[[1]], '[_]'), '[[', 1)))
afile2$chain1 <- unlist(lapply(strsplit(afile2[[1]], '[_]'), '[[', 2))
afile2$chain2 <- unlist(lapply(strsplit(afile2[[2]], '[_]'), '[[', 2))

temp_pdb <- unique(afile2[[3]])

for(mm in 1:length(cutoff)){

	store_dir <- paste0(data_dir,'networks_',cutoff[mm])
	if(!dir.exists(store_dir)){
		dir.create(store_dir)
	}

	for(k in 1:length(temp_pdb)){

		afile3 <- afile2[afile2$pdbid == temp_pdb[k], ]

		tfile <- readLines(paste0(pdbDirectory,"/",temp_pdb[k],".cif"))
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
		whf <- setdiff(seq(1,length(tfile)), wh)
		fields <- trimws(tfile[whf])

		tfile1 <- trimws(tfile[wh])
		tfile2 <- read.table(textConnection(tfile1), sep='', colClasses = "character")

		# chain id is author defined and not CIF defined
		chainPosition <- which(fields == "_atom_site.auth_asym_id")#_atom_site.label_asym_id
		chain <- tfile2[[chainPosition]]
		chain[is.na(chain)] <- "NA"

		#for each complex from this pdbid
		for(j in 1:length(afile3[[1]])){

			#filter using chain
			wh <- which(chain == as.character(afile3$chain1[j]) | chain == as.character(afile3$chain2[j]))
			tfile3 <- tfile2[wh, ]

			#filter using model number. keep only the first model
			modelPosition <- which(fields == "_atom_site.pdbx_PDB_model_num")
			mdl <- unique(tfile3[[modelPosition]])
			wh <- which(tfile3[[modelPosition]] == mdl[1])
			tfile4 <- tfile3[wh, ]

			# extract coordinates for only the heavy atoms (S, O, C, N)
			atomPosition <- which(fields == "_atom_site.type_symbol")
			lineID2 <- tfile4[[atomPosition]]
			wh <- which(lineID2 == "C" | lineID2 == "S" | lineID2 == "O" | lineID2 == "N")
			tfile6 <- tfile4[wh, ]

			# keep only ATOM coordinates
			wh <- which(tfile6[[1]] == "ATOM")
			tfile6 <- tfile6[wh,]
			seqPosition <- which(fields == "_atom_site.label_seq_id")#CIF based
			aaPosition <- which(fields == "_atom_site.label_comp_id") # get the AA symbol column

			### transformation into single chain ###########################
			# get the receptor and ligand information
			temp <- tfile6
			chains <- temp[[chainPosition]]
			uchain <- unique(chains)
			chain1 <- which(chains %in% afile3$chain1[j])
			chain2 <- which(chains %in% afile3$chain2[j])

			# original sequences
			q1 <- temp[[seqPosition]][chain1]
			q2 <- temp[[seqPosition]][chain2]
			temp[chainPosition] <- rep('Z', length(temp[[1]]))
			counter <- 1
			pointer0 <- temp[[seqPosition]][1]
			new_seq <- c(1)

			for(i in 2:length(temp[[1]])){

				pointer1 <- temp[[seqPosition]][i]

				if(pointer1 != pointer0){
					counter <- counter+1
				}

				pointer0 <- pointer1
				new_seq <- c(new_seq, counter)
			}

			temp[seqPosition] <- new_seq

			# extract chain positions of the two proteins
			temp1 <- temp[chain1, ]
			temp2 <- temp[chain2, ]
			p1 <- temp1[[seqPosition]]
			p2 <- temp2[[seqPosition]]
			a1 <- temp1[[aaPosition]]
			a2 <- temp2[[aaPosition]]

			#call to create networks
			seqq <- temp[[seqPosition]]
			wh <- which(fields == "_atom_site.Cartn_x")
			xcoord <- temp[[wh]]
			wh <- which(fields == "_atom_site.Cartn_y")
			ycoord <- temp[[wh]]
			wh <- which(fields == "_atom_site.Cartn_z")
			zcoord <- temp[[wh]]

			fname <- paste0(temp_pdb[k],'_',afile3$chain1[j],'_',afile3$chain2[j])
			uniqueids <- unique(seqq)
			lastPositions <- length(seqq)-match(unique(seqq),rev(seqq))
			xx <- pdb2net(as.character(uniqueids), as.numeric(lastPositions), as.numeric(xcoord), as.numeric(ycoord), as.numeric(zcoord), as.numeric(cutoff[mm]))


			Data <- data.frame(x=xx[[1]][1:xx[[4]]], y=xx[[2]][1:xx[[4]]])
			fwrite(Data,paste0(store_dir,'/',fname,'.txt'), quote=FALSE, sep='\t', col.names=FALSE, row.names=FALSE)
			
			pp1 <- data.frame(original=q1, new=p1, AA=a1)
			pp2 <- data.frame(original=q2, new=p2, AA=a2)

			fwrite(pp1,paste0(store_dir,'/',fname,'.chain1'), quote=FALSE, sep='\t', row.names=FALSE)
			fwrite(pp2,paste0(store_dir,'/',fname,'.chain2'), quote=FALSE, sep='\t', row.names=FALSE)

		}

		cat('For cutoff:',cutoff[mm],' :chain ',k, ' of ', length(temp_pdb), ' done\n')

	}

}

