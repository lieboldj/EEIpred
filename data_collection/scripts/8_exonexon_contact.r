##############################################################################################
# Purpose: exon pairs of the interacting interfaces of protein chains
##############################################################################################

rm(list=ls())
library(data.table)
library(plyr)
library(ggplot2)
library(stringr)
library(igraph)

#####################################################################
# change datapaths here
data_dir <- "../"
store_dir <- paste0(data_dir, "CONTACT") # name of the dataset
if(!dir.exists(store_dir)){
	dir.create(store_dir)
}
ensembl_out <- paste0(data_dir, "uniprot_pdb_Ensembl_final.txt")
mapDirectory <- paste0(data_dir, 'uniprot_EnsemblExonPDB_map')

# set cutoff in A between two interacting amino acids
cutoff <- c(6)#c(4, 5, 6, 7, 8)
#####################################################################

cmx <- fread(ensembl_out, header=TRUE)
cmx$cid1 <- paste0(cmx$PROTEIN1,'_',tolower(cmx$ID),'_',cmx$CHAIN1)
cmx$cid2 <- paste0(cmx$PROTEIN2,'_',tolower(cmx$ID),'_',cmx$CHAIN2)
upro <- union(cmx$PROTEIN1, cmx$PROTEIN2)
updb <- unique(cmx$ID)

## filter to only retain complexes present in the final exon maps

allmaps <- list.files(mapDirectory, full.names=TRUE)
##---which map files have pdb mapping according to sifts
to_keep <- c()
for(i in 1:length(allmaps)){
	tempd <- data.table::fread(allmaps[i])
    tempd1 <- tempd[tempd$PDBResNumCIF != '-', ]

	if(nrow(tempd1) != 0){
		to_keep <- c(to_keep, i)
	}
}
allmaps <- allmaps[to_keep]
shredmaps <- unlist(lapply(strsplit(basename(allmaps), '[.]'), '[[', 1))
wh1 <- which(cmx$cid1 %in% shredmaps)
wh2 <- which(cmx$cid2 %in% shredmaps)
wh <- intersect(wh1, wh2)
cmx_f <- cmx[wh,] 


####--------------------------------------------------------------------------------------------------


for(uu in 1:length(cutoff)){

	net_dir <- paste0('../../data/networks_',cutoff[uu])

	protein1 <- c()
	protein2 <- c()
	exon1_len <- c()
	exon2_len <- c()
	exon1_cov <- c()
	exon2_cov <- c()
	ex1 <- c()
	ex2 <- c()

	chainn1 <- c()
	chainn2 <- c()
	exon1n <- c()
	exon2n <- c()
	iaa1 <- c()
	iaa2 <- c()
	notp <- c()

	##--- Three are two cases where the two exons interact but the corresponding PDB AA positions are not mapped by the SIFTS 
	## uniprot to PDB mapping

	for(k in 1:length(cmx_f[[1]])){

		t1 <- tolower(cmx_f$ID[k])
		u1 <- cmx_f$PROTEIN1[k]
		u2 <- cmx_f$PROTEIN2[k]
		c1 <- cmx_f$CHAIN1[k]
		c2 <- cmx_f$CHAIN2[k]
		p1 <- cmx_f$cid1[k]
		p2 <- cmx_f$cid2[k]
		cmx_name <- paste0(t1,'_',c1,'_',c2)
		temp1 <- fread(list.files(mapDirectory, pattern=paste0('^',u1,'_',t1,'_',c1), full.names=TRUE), sep='\t', header=TRUE)
		temp2 <- fread(list.files(mapDirectory, pattern=paste0('^',u2,'_',t1,'_',c2), full.names=TRUE), sep='\t', header=TRUE)
		
		# only consider the resolved residues
		temp1 <- temp1[temp1$PDBResNumCIF != '-', ]
		temp2 <- temp2[temp2$PDBResNumCIF != '-', ]
		if(nrow(temp1) == 0 | nrow(temp2) == 0){ # if at least one mapping file has no resolved residues
			notp <- c(notp, k)
			next
		}
		temp1_ex <- unique(temp1$EXON)
		temp2_ex <- unique(temp2$EXON)
		loop1 <- length(temp1_ex)
		loop2 <- length(temp2_ex)
		if(file.exists(paste0(net_dir,'/',cmx_name,'.chain1'))){ # checking whether the name of network is saved as representing the first chain1 first or the second chain first
			cmx_name <- cmx_name
		}else{
			cmx_name <- paste0(t1,'_',c2,'_',c1)
		}
		chain1 <- unique(fread(paste0(net_dir,'/',cmx_name,'.chain1'), header=TRUE))
		chain2 <- unique(fread(paste0(net_dir,'/',cmx_name,'.chain2'), header=TRUE))
		cmx_net <- fread(paste0(net_dir,'/',cmx_name,'.txt'))

		for(i in 1:loop1){
			temp11 <- temp1[temp1$EXON == temp1_ex[i], ]
			est1 <- min(as.numeric(temp11$PDBResNumCIF))
			eed1 <- max(as.numeric(temp11$PDBResNumCIF))
			seq1 <- seq(est1, eed1) # seq nums defined by CIF
			seqq1 <- unique(chain1[which(chain1$original %in% seq1),][[2]]) # get seq nums defined by me


			for(j in 1:loop2){
				temp22 <- temp2[temp2$EXON == temp2_ex[j], ]
				est2 <- min(as.numeric(temp22$PDBResNumCIF))
				eed2 <- max(as.numeric(temp22$PDBResNumCIF))
				seq2 <- seq(est2, eed2) # seq nums defined by CIF
				seqq2 <- unique(chain2[which(chain2$original %in% seq2),][[2]]) # get seq nums defined by me

				if(length(seqq1) != 0 & length(seqq2) != 0){ # if both exons have at least one amino acid resolved

					wh1 <- which((cmx_net$V1 %in% seqq1) & (cmx_net$V2 %in% seqq2))
					wh2 <- which((cmx_net$V2 %in% seqq1) & (cmx_net$V1 %in% seqq2))
					wh <- union(wh1, wh2)

					if(length(wh) != 0){ # if the two exons interact

						if((length(wh1) != 0) & (length(wh2) != 0)){
							temp_net1 <- cmx_net[wh1, ]
							temp_net2 <- cmx_net[wh2, ]
							tt1 <- temp_net2[[2]]
							temp_net2[[2]] <- temp_net2[[1]]
							temp_net2[[1]] <- tt1
							temp_net <- rbind(temp_net1, temp_net2)
						}else if(length(wh1) != 0){
							temp_net <- cmx_net[wh1, ]
						}else if(length(wh2) != 0){
							temp_net <- cmx_net[wh2, ]
							tt1 <- temp_net[[2]]
							temp_net[[2]] <- temp_net[[1]]
							temp_net[[1]] <- tt1
						}

						allnodes <- union(temp_net[[1]], temp_net[[2]])
						## save all nodes before updating temp_net for later use

						## store the PDBResNumCIF info -------
						ch1 <- c()
						for(m in 1:length(temp_net[[1]])){
							ch1 <- c(ch1, chain1[which(chain1$new == temp_net[[1]][m]),]$original)
						}

						ch2 <- c()
						for(m in 1:length(temp_net[[2]])){
							ch2 <- c(ch2, chain2[which(chain2$new == temp_net[[2]][m]),]$original)
						}


						## ---- remove the PDBResNumCIF for which there are no UNIPROT-based AA positions
						sch1 <- c()
						for(ii in 1:length(ch1)){
							xx <- ch1[ii]
							xxx <- which(temp1$PDBResNumCIF == xx)
							if(length(xxx) == 0){
								sch1 <- c(sch1, xx)
							}
						}

						## ---- remove the PDBResNumCIF for which there are no UNIPROT-based AA positions
						sch2 <- c()
						for(ii in 1:length(ch2)){
							xx <- ch2[ii]
							xxx <- which(temp2$PDBResNumCIF == xx)
							if(length(xxx) == 0){
								sch2 <- c(sch2, xx)
							}
						}

						## convert sch from CIF to new IDs
						ssch1 <- c()
						for(m in 1:length(sch1)){
							ssch1 <- c(ssch1, chain1[which(chain1$original == sch1[m]),]$new)
						}

						ssch2 <- c()
						for(m in 1:length(sch2)){
							ssch2 <- c(ssch2, chain2[which(chain2$original == sch2[m]),]$new)
						}

						tokeep1 <- setdiff(temp_net[[1]], ssch1)
						tokeep2 <- setdiff(temp_net[[2]], ssch2)

						wh1 <- which(temp_net[[1]] %in% tokeep1)
						wh2 <- which(temp_net[[2]] %in% tokeep2)
						wh <- intersect(wh1, wh2)

						temp_net <- temp_net[wh, ]


						##---- now catch whether now temp_net is empty ----
						if(nrow(temp_net) != 0){

							## store the PDBResNumCIF info
							ch1 <- c()
							for(m in 1:length(temp_net[[1]])){
								ch1 <- c(ch1, chain1[which(chain1$new == temp_net[[1]][m]),]$original)
							}

							ch2 <- c()
							for(m in 1:length(temp_net[[2]])){
								ch2 <- c(ch2, chain2[which(chain2$new == temp_net[[2]][m]),]$original)
							}

							tt1 <- c()
							for(ii in 1:length(ch1)){
								xx <- ch1[ii]
								xxx <- which(temp1$PDBResNumCIF == xx)
								tt1 <- c(tt1, temp1[xxx,]$UNIPROT_SEQ_NUM)
							}
							
							tt2 <- c()
							for(ii in 1:length(ch2)){
								xx <- ch2[ii]
								xxx <- which(temp2$PDBResNumCIF == xx) 
								tt2 <- c(tt2, temp2[xxx,]$UNIPROT_SEQ_NUM)
							}

							## store for actaul AA interactions
							exon1n <- c(exon1n, rep(temp1_ex[i], length(tt1)))
							exon2n <- c(exon2n, rep(temp2_ex[j], length(tt2)))
							iaa1 <- c(iaa1, tt1)
							iaa2 <- c(iaa2, tt2)
							chainn1 <- c(chainn1, rep(p1, length(tt1)))
							chainn2 <- c(chainn2, rep(p2, length(tt2)))

						} 


						# store info
						protein1 <- c(protein1, p1)
						protein2 <- c(protein2, p2)
						exon1_len <- c(exon1_len, length(seqq1)) # length based on resolved 3D structure
						exon2_len <- c(exon2_len, length(seqq2)) # length based on resolved 3D structure
						exon1_cov <- c(exon1_cov, length(intersect(allnodes, seqq1)))
						exon2_cov <- c(exon2_cov, length(intersect(allnodes, seqq2)))
						ex1 <- c(ex1, temp1_ex[i])
						ex2 <- c(ex2, temp2_ex[j])

					}else{ # if no interaction
						# store info
						protein1 <- c(protein1, p1)
						protein2 <- c(protein2, p2)

						exon1_len <- c(exon1_len, length(seqq1)) # length based on resolved 3D structure
						exon2_len <- c(exon2_len, length(seqq2)) # length based on resolved 3D structure

						exon1_cov <- c(exon1_cov, 0)
						exon2_cov <- c(exon2_cov, 0)

						ex1 <- c(ex1, temp1_ex[i])
						ex2 <- c(ex2, temp2_ex[j])
					}
				}
			}
		}
		cat('For cutoff:',cutoff[uu],' : complex', k, ' of ', length(cmx_f[[1]]), ' done\n')
	}


	all_exon_pairs <- data.frame(protein1=protein1, protein1=protein2, exon1=ex1, exon2=ex2, exon1_length=as.numeric(exon1_len),
		exon2_length=as.numeric(exon2_len), exon1_coverage=exon1_cov, exon2_coverage=exon2_cov)

	all_exon_pairs$exon1_coverage_percent <- round((all_exon_pairs$exon1_coverage/all_exon_pairs$exon1_length)*100, 2)
	all_exon_pairs$exon2_coverage_percent <- round((all_exon_pairs$exon2_coverage/all_exon_pairs$exon2_length)*100, 2)
	all_exon_pairs$jaccard_percent <- round(((all_exon_pairs$exon1_coverage+all_exon_pairs$exon2_coverage)/(all_exon_pairs$exon1_length+all_exon_pairs$exon2_length))*100, 2)
	fwrite(all_exon_pairs, paste0(store_dir,'/exon_pairs',cutoff[uu],'.txt'), sep='\t', row.names=FALSE, quote=FALSE)

	## exact interacting AA info
	aa_data <- data.frame(chain1=chainn1, chain2=chainn2, exon1=exon1n, exon2=exon2n, exon1_UNIPROT_SEQ_NUM=iaa1, exon2_UNIPROT_SEQ_NUM=iaa2)
	fwrite(aa_data, paste0(store_dir,'/aa_interactions',cutoff[uu],'.txt'), sep='\t', row.names=FALSE, quote=FALSE)

	## distribution of the edges based on the jaccard ########
	temp_exon <- fread( paste0(store_dir,'/exon_pairs',cutoff[uu],'.txt'), sep='\t', header=TRUE)

	int_exon <- temp_exon[temp_exon$exon1_coverage > 0, ]
	nint_exon <- temp_exon[temp_exon$exon1_coverage == 0, ]
	fwrite(int_exon, paste0(store_dir,'/int_exon_pairs',cutoff[uu],'.txt'), sep='\t', row.names=FALSE, quote=FALSE)
	fwrite(nint_exon, paste0(store_dir,'/nint_exon_pairs',cutoff[uu],'.txt'), sep='\t', row.names=FALSE, quote=FALSE)


}

