##############################################################################################
# Purpose: save different contact-based networks
##############################################################################################

rm(list=ls())
library(Rcpp)

#####################################################################
# change datapaths here
data_dir <- "../"
in_dir <- paste0(data_dir, "CONTACT") # name of the dataset
store_dir <- paste0(data_dir, "CONTACT_networks") # name of the dataset
if(!dir.exists(store_dir)){
	dir.create(store_dir)
}
# set cutoff in A between two interacting amino acids
cutoff <- c(6)#c(4, 5, 6, 7, 8)
# set the number of amino acids which at least have to be closer than the cutoff
num_aa <- c(1)#,3,5,7,9)

#####################################################################

cppFunction("List filtereei(CharacterVector ex1, CharacterVector ex2, CharacterVector exon1, CharacterVector exon2, NumericVector allaa){

    int loop1 = ex1.size();
    int loop2 = exon1.size();
    NumericVector allpositions;

    for(int k=0; k<loop1; k++){

		NumericVector loopx;
		NumericVector tallaa;

        for(int j=0; j<loop2; j++){

            if((ex1[k] == exon1[j]) & (ex2[k] == exon2[j])){
                loopx.push_back(j);
                tallaa.push_back(allaa[j]);
            }

            if((ex1[k] == exon2[j]) & (ex2[k] == exon1[j])){
                loopx.push_back(j);
                tallaa.push_back(allaa[j]);
            }
        }

        int loop3 = tallaa.size();
        int MAX = 0;
        int position = 0;
        for(int j=0; j<loop3; j++){
        	if(tallaa[j] > MAX){
        		MAX = tallaa[j];
        		position = loopx[j];
        	}
            //Rcout << MAX << std::endl;
        }
        //Rcout << loop3 << std::endl;

        allpositions.push_back(position);
    }

    List L = List::create(allpositions);
	return L;
  
}")


for(k in 1:length(cutoff)){

	temp <- data.table::fread(paste0(in_dir,'/int_exon_pairs',cutoff[k],'.txt'))

	for(j in 1:length(num_aa)){

		wh1 <- which(temp$exon1_coverage >= num_aa[j])
		wh2 <- which(temp$exon2_coverage >= num_aa[j])
		temp1 <- temp[intersect(wh1, wh2), ]
		temp1$allAA <- temp1$exon1_coverage+temp1$exon2_coverage
		gnet <- igraph::as_data_frame(igraph::simplify(igraph::graph_from_data_frame(temp1[,c(3,4)], directed=FALSE)))
		tempf <- filtereei(gnet[[1]], gnet[[2]], temp1$exon1, temp1$exon2, temp1$allAA)
		ids <- tempf[[1]]+1
		tempff <- temp[ids, ]
		data.table::fwrite(tempff, paste0(store_dir,'/CONTACT_net_',cutoff[k],'_',num_aa[j],'_positives.txt'), sep='\t', row.names=FALSE, quote=FALSE)

	}
}


for(k in 1:length(cutoff)){

    temp <- data.table::fread(paste0(in_dir,'/nint_exon_pairs',cutoff[k],'.txt'))

    for(j in 1:length(num_aa)){
        tempff <- temp
        data.table::fwrite(tempff, paste0(store_dir,'/CONTACT_net_',cutoff[k],'_',num_aa[j],'_negatives.txt'), sep='\t', row.names=FALSE, quote=FALSE)

    }
}


