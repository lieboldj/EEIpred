##############################################################################################
# Purpose: generate protonated files of the PDB using the program reduce to use in the dmasif method
##############################################################################################

rm(list=ls())
library(data.table)
library(stringr)
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("bio3d")
library(bio3d)

data_dir <- "../"
indir <- paste0(data_dir, 'PDB_chains')
outdir <- paste0(data_dir, 'PDB_chains_protonated')
# might change reduce to ./reduce if it is not in the path but in the current directory
reduce_cmd <- "reduce"
# MAKE SURE TO INSTALL REDUCE PRIOR TO RUNNING THIS SCRIPT


if (!dir.exists(outdir))
	dir.create(outdir, recursive=TRUE)

allfiles <- list.files(indir, full.names=TRUE)

for(k in 1:length(allfiles)){

	ba <- basename(allfiles[k])
	if (file.exists(paste0(outdir,'/',ba)))
		next
	
	cmd <- paste(reduce_cmd, ' -build -Quiet',allfiles[k],'>',paste0(outdir,'/',ba))
	system(cmd)

	tfile <- readLines(paste0(outdir,'/',ba))
	lineID <- word(tfile, 1)
	wh <- which(lineID == 'ATOM')
	tfile1 <- trimws(tfile[wh])
	tfile2 <- unlist(lapply(tfile1, function(x) substr(x,1,78)))
	# tfile3 <- read.table(textConnection(tfile2), sep='', colClasses = "character")
	
	ncc1 <- trimws(unlist(lapply(tfile2, function(x) substr(x,22,22))))## chain identifier
	atm1 <- seq(1,length(tfile2), 1) ## atom serial number
	atms <- trimws(unlist(lapply(tfile2, function(x) substr(x,13,16)))) ## atom names
	ress <- trimws(unlist(lapply(tfile2, function(x) substr(x,18,20)))) ## residue name
	reso <- trimws(unlist(lapply(tfile2, function(x) substr(x,23,26)))) ## residue sequence number
	hatm <- trimws(unlist(lapply(tfile2, function(x) substr(x,77,78)))) ## element symbol
	oflag <- trimws(unlist(lapply(tfile2, function(x) substr(x,55,60)))) ## occupancy
	bflag <- trimws(unlist(lapply(tfile2, function(x) substr(x,61,66)))) ## temperature factor
	coordx <- trimws(unlist(lapply(tfile2, function(x) substr(x,31,38)))) ## X coordinate
	coordy <- trimws(unlist(lapply(tfile2, function(x) substr(x,39,46)))) ## Y coordinate
	coordz <- trimws(unlist(lapply(tfile2, function(x) substr(x,47,54)))) ## Z coordinate
	coord <- data.frame(coordx, coordy, coordz)
	etype <- trimws(unlist(lapply(tfile2, function(x) substr(x,1,4))))## Entry type

	write.pdb(file=paste0(outdir,'/',ba),
			type=etype,
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
	cat('PDB', k, 'of',length(allfiles), 'done\n')
}

