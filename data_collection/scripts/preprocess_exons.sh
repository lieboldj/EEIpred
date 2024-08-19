#!/usr/bin/sh

# select only exons
awk -F '\t' '$3 == "exon"' ../Homo_sapiens.GRCh38.111.gtf > $1
#../../data/Homo_sapiens.GRCh38.111_exons.tmp
#awk -F '\t' '$3 == "gene"' ../../data/Homo_sapiens.GRCh38.111.gtf > ../../data/Homo_sapiens.GRCh38.111_genes.gtf
awk -F '\t' '$3 == "CDS"' ../Homo_sapiens.GRCh38.111.gtf > $2
#../../data/Homo_sapiens.GRCh38.111_cds.tmp

