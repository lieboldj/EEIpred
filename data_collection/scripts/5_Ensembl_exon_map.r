##############################################################################################
# Purpose: download Ensembl data
##############################################################################################

rm(list=ls())
library(data.table)
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("biomaRt")
library(biomaRt)
library(seqinr)

substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

cleanSeq <- function(x){
	last_flag <- substrRight(x, 1)
	if(last_flag == "*"){
		slen <- nchar(x)-1
		x <- substr(x, 1, slen)
	}
	return(x)
}

# download GTF file from Ensembl
# downloaded 25.04.2024
#########################################################
# change variables here for other versions or species
data_dir <- "../"
process_file <- paste0(data_dir, "processed_complex_final.txt")
xx <- "Homo_sapiens.GRCh38.111.gtf.gz"
dataset_ensemble <- "hsapiens_gene_ensembl"
data_abbr <- "HS"
exons_tmp <- gsub(".gtf.gz", "_exons.tmp", xx)
exons_tmp <- paste0(data_dir,exons_tmp)
cds_tmp <- gsub(".gtf.gz", "_cds.tmp", xx)
cds_tmp <- paste0(data_dir,cds_tmp)

ensemble_exon_dir <- "Ensembl_exons_T.txt"
ensemble_exon_dir <- paste0(data_dir,ensemble_exon_dir)
ensemble_cds_dir <- "Ensembl_exon_cds.txt"
ensemble_cds_dir <- paste0(data_dir,ensemble_cds_dir)

ensembl_out <- "uniprot_pdb_Ensembl_final.txt"
ensembl_out <- paste0(data_dir, ensembl_out)

priassm <- 'Homo_sapiens.GRCh38.dna.primary_assembly.fa'
genome <- seqinr::read.fasta(priassm)

# map exons to uniprot sequences #####
store_dir <- paste0(data_dir, 'uniprot_Ensembl_Exon_map')
if(dir.exists(store_dir)){
	unlink(store_dir, recursive=TRUE)
}
dir.create(store_dir)
#######################################################

file_path <- paste0(data_dir, xx)
print(file_path)
if(!file.exists(file_path)){
	print("if not worked")
	system(paste0("wget -O ", data_dir, xx," http://ftp.ensembl.org/pub/current_gtf/homo_sapiens/",xx))
	system(paste0("gunzip ", data_dir, xx))
}
# preprocess
if (!(file.exists(exons_tmp) & file.exists(cds_tmp))) {
	prepro <- paste0('sh ./preprocess_exons.sh ', exons_tmp, ' ', cds_tmp)
	system(prepro)
}

exons <- fread(exons_tmp,sep='\t')
cds <- fread(cds_tmp,sep='\t')

xxs <- strsplit(exons$V9, '[;]')
## gene_id

wh1 <- lapply(xxs, function(x) which(x %like% 'gene_id'))
gene_id1 <- mapply(function(x, y) x[y], xxs, wh1)
gene_id <- gsub('\\"', "", unlist(lapply(strsplit(trimws(gene_id1),'\\s+'), '[[', 2)))

## transcript_id
wh1 <- lapply(xxs, function(x) which(x %like% 'transcript_id'))
transcript_id1 <- mapply(function(x, y) x[y], xxs, wh1)
transcript_id <- gsub('\\"', "", unlist(lapply(strsplit(trimws(transcript_id1),'\\s+'), '[[', 2)))

## exon_number
wh1 <- lapply(xxs, function(x) which(x %like% 'exon_number'))
exon_number1 <- mapply(function(x, y) x[y], xxs, wh1)
exon_number <- gsub('\\"', "", unlist(lapply(strsplit(trimws(exon_number1),'\\s+'), '[[', 2)))

## transcript_biotype
wh1 <- lapply(xxs, function(x) which(x %like% 'transcript_biotype'))
transcript_biotype1 <- mapply(function(x, y) x[y], xxs, wh1)
transcript_biotype <- gsub('\\"', "", unlist(lapply(strsplit(trimws(transcript_biotype1),'\\s+'), '[[', 2)))

## exon_id
wh1 <- lapply(xxs, function(x) which(x %like% 'exon_id'))
exon_id1 <- mapply(function(x, y) x[y], xxs, wh1)
exon_id <- gsub('\\"', "", unlist(lapply(strsplit(trimws(exon_id1),'\\s+'), '[[', 2)))

exons$gene_id <- gene_id
exons$transcript_id <- transcript_id
exons$transcript_biotype <- transcript_biotype
exons$exon_number <- exon_number
exons$exon_id <- exon_id
#print(exons)
exons_f <- exons[,-9]
fwrite(exons_f, ensemble_exon_dir, sep='\t', quote=FALSE, row.names=FALSE)
exons_f <- fread(ensemble_exon_dir, header=TRUE)


##### CDS
xxs <- strsplit(cds$V9, '[;]')
# gene_id
wh1 <- lapply(xxs, function(x) which(x %like% 'gene_id'))
gene_id1 <- mapply(function(x, y) x[y], xxs, wh1)
gene_id <- gsub('\\"', "", unlist(lapply(strsplit(trimws(gene_id1),'\\s+'), '[[', 2)))

# transcript_id
wh1 <- lapply(xxs, function(x) which(x %like% 'transcript_id'))
transcript_id1 <- mapply(function(x, y) x[y], xxs, wh1)
transcript_id <- gsub('\\"', "", unlist(lapply(strsplit(trimws(transcript_id1),'\\s+'), '[[', 2)))

# exon_number
wh1 <- lapply(xxs, function(x) which(x %like% 'exon_number'))
exon_number1 <- mapply(function(x, y) x[y], xxs, wh1)
exon_number <- gsub('\\"', "", unlist(lapply(strsplit(trimws(exon_number1),'\\s+'), '[[', 2)))

# transcript_biotype
wh1 <- lapply(xxs, function(x) which(x %like% 'transcript_biotype'))
transcript_biotype1 <- mapply(function(x, y) x[y], xxs, wh1)
transcript_biotype <- gsub('\\"', "", unlist(lapply(strsplit(trimws(transcript_biotype1),'\\s+'), '[[', 2)))

cds$gene_id <- gene_id
cds$transcript_id <- transcript_id
cds$transcript_biotype <- transcript_biotype
cds$exon_number <- exon_number
cds$nt_len <- (cds$V5-cds$V4)+1

cds_f <- cds[,-9]

fwrite(cds_f, ensemble_cds_dir, sep='\t', quote=FALSE, row.names=FALSE)
cds_f <- fread(ensemble_cds_dir, header=TRUE)

# get the uniprot emsembl mapping ##### USE OF BIOMART #########################
ensembl <- useEnsembl(biomart = "genes", dataset = dataset_ensemble, mirror='useast') # useast uswest asia
# all data info
attrs <- c('ensembl_gene_id', 'ensembl_transcript_id', 'ensembl_peptide_id',
	'transcript_biotype', 'uniprotswissprot')
####################################################################################

# for each uniprot ....
# all uniprot ids
uniprot_pdb <- fread(process_file, sep='\t', header=TRUE)

# correct for NA chain recognized as a missing value
uniprot_pdb[is.na(uniprot_pdb)] <- "NA"
# remove the complexes where the proteins map to the same chain as per SIFTS mapping


to_keep <- c()
for(i in 1:length(uniprot_pdb[[1]])){
	if(uniprot_pdb$CHAIN1[i] != uniprot_pdb$CHAIN2[i]){
		to_keep <- c(to_keep, i)
	}
}
# remove the complexes where the proteins map to the same chain as per SIFTS mapping

uniprot_pdb <- uniprot_pdb[to_keep, ]
upro <- union(uniprot_pdb$PROTEIN1, uniprot_pdb$PROTEIN2)
updb <- unique(uniprot_pdb$ID)
##-- 13,818 out of 13,967 protein pairs remain
# get the uniprot sequences
seq_list <- paste0(data_dir, 'uniprot_sequences_', data_abbr, '.txt')

uniprot_seqs <- read.fasta(seq_list, seqtype="AA", whole.header=TRUE)
uniprotids1 <- uniprot_pdb$PROTEIN1
uniprotids2 <- uniprot_pdb$PROTEIN2
uniprotids <- union(uniprotids1, uniprotids2)
alld <- getBM(attributes=attrs, filters='uniprotswissprot', values=uniprotids, mart=ensembl)

bh_transcripts1 <- rep('',length(uniprotids1))

# countl <- c()
countl <- rep(0, length(uniprotids1))
count_gap <- rep(0, length(uniprotids1))
iden <- rep(0,length(uniprotids1))

for(k in 1:length(uniprotids1)){
	## 3601 --> global alignment giving some error
	# if(k == 3601){next}

	# length of the uniprot seq
	uni_seq <- getLength(uniprot_seqs[uniprotids1[k]])
	alldata <- alld[alld$uniprotswissprot == uniprotids1[k], ]

	# only protein coding
	alldata1 <- alldata[alldata$transcript_biotype == 'protein_coding', ]
	all_transcripts <- alldata1$ensembl_transcript_id

	if(!identical(all_transcripts, character(0))){
		temp_seq <- biomaRt::getSequence(id=all_transcripts, type='ensembl_transcript_id', seqType='peptide', mart=ensembl)

		for(j in 1:length(all_transcripts)){

			temp_cds <- cds_f[cds_f$transcript_id == all_transcripts[j], ]
			
			if(nrow(temp_cds) > 0){
				dna_strand <- temp_cds$V7[[1]][1]
				temp_sum <- sum(temp_cds$nt_len)/3
				flags <- 0
				if(temp_sum == uni_seq){
					flags <- flags+1
					
					## global alignment
					toalign1 <- temp_seq$peptide[which(temp_seq$ensembl_transcript_id == all_transcripts[j])] ## protein seqeunce of first transcript
					toalign1 <- cleanSeq(toalign1)

					# temps <- seqinr::getSequence(genome[paste0('chr',temp_cds[[1]][1])])[[1]]
					temps <- seqinr::getSequence(genome[temp_cds[[1]][1]])[[1]]

					allseq <- c()
					for(i in 1:length(temp_cds[[1]])){
						start <- temp_cds$V4[i]
						end <- temp_cds$V5[i]
						tempseq <- temps[start:end]
						if(dna_strand == "-"){
							tempseq <- rev(seqinr::comp(tempseq))
						}
						allseq <- c(allseq, tempseq)
					}
					
					toalign2 <- toupper(paste(seqinr::translate(allseq),collapse=''))

					xxx <- Biostrings::pairwiseAlignment(toalign1, toalign2,
			                     substitutionMatrix = "BLOSUM62", 
			                     gapOpening = -2,
			                     gapExtension = -8, 
			                     scoreOnly = FALSE)

					iso1 <- as.vector(stringr::str_split_fixed(as.character(Biostrings::pattern(xxx)), pattern='', n=nchar(as.character(Biostrings::pattern(xxx)))))
					iso2 <- as.vector(stringr::str_split_fixed(as.character(Biostrings::subject(xxx)), pattern='', n=nchar(as.character(Biostrings::subject(xxx)))))

					gap1 <- length(which(iso1 == '-')) # number of gaps in isoform1
					gap2 <- length(which(iso2 == '-')) # number of gaps in isoform 2
					gap <- gap1+gap2
					count_gap[k] <- gap
					iden[k] <- 100-Biostrings::pid(xxx, type='PID3')
					if(iden[k] == 0 && gap == 0){
						bh_transcripts1[k] <- all_transcripts[j]
						break
					}
				}
			}
		}
		countl[k] <- flags
	}
	cat('Protein',k,' of ', length(uniprotids1), ' done\n')
}

countl1 <- count_gap
idenl1 <- iden

bh_transcripts2 <- rep('',length(uniprotids2))
countl <- c()
count_gap <- rep(0, length(uniprotids2))
iden <- rep(0,length(uniprotids2))

for(k in 1:length(uniprotids2)){

	# if(k == 2415 | k == 3841 | k==4108){next}
	# length of the uniprot seq
	uni_seq <- getLength(uniprot_seqs[uniprotids2[k]])
	alldata <- alld[alld$uniprotswissprot == uniprotids2[k], ]

	# only protein coding
	alldata1 <- alldata[alldata$transcript_biotype == 'protein_coding', ]
	all_transcripts <- alldata1$ensembl_transcript_id

	if(!identical(all_transcripts, character(0))){
		temp_seq <- biomaRt::getSequence(id=all_transcripts, type='ensembl_transcript_id', seqType='peptide', mart=ensembl)

		for(j in 1:length(all_transcripts)){

			temp_cds <- cds_f[cds_f$transcript_id == all_transcripts[j], ]
			
			if(nrow(temp_cds) > 0){
				dna_strand <- temp_cds$V7[[1]][1]
				temp_sum <- sum(temp_cds$nt_len)/3
				flags <- 0
				if(temp_sum == uni_seq){
					flags <- flags+1

					## global alignment
					toalign1 <- temp_seq$peptide[which(temp_seq$ensembl_transcript_id == all_transcripts[j])] ## protein seqeunce of first transcript
					toalign1 <- cleanSeq(toalign1)

					# temps <- seqinr::getSequence(genome[paste0('chr',temp_cds[[1]][1])])[[1]]
					temps <- seqinr::getSequence(genome[temp_cds[[1]][1]])[[1]]

					allseq <- c()
					for(i in 1:length(temp_cds[[1]])){
						start <- temp_cds$V4[i]
						end <- temp_cds$V5[i]
						tempseq <- temps[start:end]
						if(dna_strand == "-"){
							tempseq <- rev(seqinr::comp(tempseq))
						}
						allseq <- c(allseq, tempseq)
					}
					
					toalign2 <- toupper(paste(seqinr::translate(allseq),collapse=''))

					xxx <- Biostrings::pairwiseAlignment(toalign1, toalign2,
			                     substitutionMatrix = "BLOSUM62", 
			                     gapOpening = -2,
			                     gapExtension = -8, 
			                     scoreOnly = FALSE)

					iso1 <- as.vector(stringr::str_split_fixed(as.character(Biostrings::pattern(xxx)), pattern='', n=nchar(as.character(Biostrings::pattern(xxx)))))
					iso2 <- as.vector(stringr::str_split_fixed(as.character(Biostrings::subject(xxx)), pattern='', n=nchar(as.character(Biostrings::subject(xxx)))))

					gap1 <- length(which(iso1 == '-')) # number of gaps in isoform1
					gap2 <- length(which(iso2 == '-')) # number of gaps in isoform 2
					gap <- gap1+gap2
					count_gap[k] <- gap
					iden[k] <- 100-Biostrings::pid(xxx, type='PID3')
					if(iden[k] == 0 && gap == 0){
						bh_transcripts2[k] <- all_transcripts[j]
						break
					}
				}
			}
		}
		countl[k] <- flags
	}
	cat('Protein',k,' of ', length(uniprotids2), ' done\n')
}


uniprot_pdb$bh_ensembl_transcript_id1 <- bh_transcripts1
uniprot_pdb$bh_ensembl_transcript_id2 <- bh_transcripts2

uniprot_pdb_final1 <- uniprot_pdb[uniprot_pdb$bh_ensembl_transcript_id1 != '', ]
uniprot_pdb_final2 <- uniprot_pdb_final1[uniprot_pdb_final1$bh_ensembl_transcript_id2 != '', ]

fwrite(uniprot_pdb_final2, '../data/uniprot_pdb_Ensembl_final.txt', sep='\t', quote=FALSE, row.names=FALSE)
uniprot_pdb_final <- data.table::fread('../data/uniprot_pdb_Ensembl_final.txt', header=TRUE)


### do the exon mapping to uniprot sequences ----------------------------------------
uniprot_uni_ids1 <- uniprot_pdb_final$PROTEIN1
transcript_uni_ids1 <- uniprot_pdb_final$bh_ensembl_transcript_id1
chain_ids1 <- uniprot_pdb_final$CHAIN1
pdb_ids <- tolower(uniprot_pdb_final$ID)

uniprot_uni_ids2 <- uniprot_pdb_final$PROTEIN2
transcript_uni_ids2 <- uniprot_pdb_final$bh_ensembl_transcript_id2
chain_ids2 <- uniprot_pdb_final$CHAIN2

# for one of the interacting protein
for(k in 1:length(transcript_uni_ids1)){

	# get the uniprot sequence
	uniprot_seq <- getSequence(uniprot_seqs[uniprot_uni_ids1[k]])[[1]]

	# get all exons for this transcript
	allexons <- exons_f[exons_f$transcript_id == transcript_uni_ids1[k], ]

	# get exon numbers from cds...denoting the coding exons for this transcript
	cexon_num <- cds_f[cds_f$transcript_id == transcript_uni_ids1[k], ]
	cexon_num <- cexon_num[order(cexon_num$exon_number), ]

	# get exons
	cexon <- allexons[allexons$exon_number %in% cexon_num$exon_number, ]

	# sort exons by genomic start
	cexon <- cexon[order(cexon$exon_number), ]

	# for each exon
	exon_entry <- rep('',length(uniprot_seq))
	exon_num_entry <- rep('',length(uniprot_seq))
	start_pos <- 1
	end_pos <- 0
	previous <- 0
	for(j in 1:length(cexon[[1]])){

		temp_pos <- (cexon_num$nt_len[j]+previous)/3

		if(temp_pos != 0){

			temp_pos1 <- floor(temp_pos)
			diff <- temp_pos-temp_pos1

			if(diff == 0){ # integer temp_pos
				end_pos <- end_pos+temp_pos1
				exon_entry[start_pos:end_pos] <- cexon$exon_id[j]
				exon_num_entry[start_pos:end_pos] <- cexon$exon_number[j]
				start_pos <- end_pos+1
				previous <- 0
			}else if(diff < 0.5){
				end_pos <- end_pos+temp_pos1
				exon_entry[start_pos:end_pos] <- cexon$exon_id[j]
				exon_num_entry[start_pos:end_pos] <- cexon$exon_number[j]
				start_pos <- end_pos+1
				previous <- 1
			}else{
				end_pos <- end_pos+temp_pos1+1
				exon_entry[start_pos:end_pos] <- cexon$exon_id[j]
				exon_num_entry[start_pos:end_pos] <- cexon$exon_number[j]
				start_pos <- end_pos+1
				previous <- -1
			}

		}
	
	}

	Data <- data.frame(UNIPROT_SEQ_NUM=seq(1,length(uniprot_seq)), UNIPROT=uniprot_seq, EXON=exon_entry, EXON_NUM=exon_num_entry)
	
	fwrite(Data, paste0(store_dir,'/',uniprot_uni_ids1[k],'_',pdb_ids[k],'_',chain_ids1[k],'.txt'), row.names=FALSE, sep='\t', quote=FALSE)
	cat('Protein',k,' of ', length(transcript_uni_ids1), ' done\n')

}


# for the other interacting protein
for(k in 1:length(transcript_uni_ids2)){

	# get the uniprot sequence
	uniprot_seq <- getSequence(uniprot_seqs[uniprot_uni_ids2[k]])[[1]]

	# get all exons for this transcript
	allexons <- exons_f[exons_f$transcript_id == transcript_uni_ids2[k], ]

	# get exon numbers from cds...denoting the coding exons for this transcript
	cexon_num <- cds_f[cds_f$transcript_id == transcript_uni_ids2[k], ]
	cexon_num <- cexon_num[order(cexon_num$exon_number), ]

	# get exons
	cexon <- allexons[allexons$exon_number %in% cexon_num$exon_number, ]

	# sort exons by genomic start
	cexon <- cexon[order(cexon$exon_number), ]

	# for each exon
	exon_entry <- rep('',length(uniprot_seq))
	exon_num_entry <- rep('',length(uniprot_seq))
	start_pos <- 1
	end_pos <- 0
	previous <- 0
	for(j in 1:length(cexon[[1]])){

		temp_pos <- (cexon_num$nt_len[j]+previous)/3

		if(temp_pos != 0){

			temp_pos1 <- floor(temp_pos)
			diff <- temp_pos-temp_pos1

			if(diff == 0){ # integer temp_pos
				end_pos <- end_pos+temp_pos1
				exon_entry[start_pos:end_pos] <- cexon$exon_id[j]
				exon_num_entry[start_pos:end_pos] <- cexon$exon_number[j]
				start_pos <- end_pos+1
				previous <- 0
			}else if(diff < 0.5){
				end_pos <- end_pos+temp_pos1
				exon_entry[start_pos:end_pos] <- cexon$exon_id[j]
				exon_num_entry[start_pos:end_pos] <- cexon$exon_number[j]
				start_pos <- end_pos+1
				previous <- 1
			}else{
				end_pos <- end_pos+temp_pos1+1
				exon_entry[start_pos:end_pos] <- cexon$exon_id[j]
				exon_num_entry[start_pos:end_pos] <- cexon$exon_number[j]
				start_pos <- end_pos+1
				previous <- -1
			}

		}
	
	}

	Data <- data.frame(UNIPROT_SEQ_NUM=seq(1,length(uniprot_seq)), UNIPROT=uniprot_seq, EXON=exon_entry, EXON_NUM=exon_num_entry)
	fwrite(Data, paste0(store_dir,'/',uniprot_uni_ids2[k],'_',pdb_ids[k],'_',chain_ids2[k],'.txt'), row.names=FALSE, sep='\t', quote=FALSE)
	cat('Protein',k,' of ', length(transcript_uni_ids2), ' done\n')

}
