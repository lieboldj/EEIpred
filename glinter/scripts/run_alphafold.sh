#! /usr/bin/bash

export GLINT_ROOT=./
export REDUCE_PATH=$GLINT_ROOT/external/reduce
export PATH=$PATH:$REDUCE_PATH
export REDUCE_HET_DICT=$REDUCE_PATH/reduce_wwPDB_het_dict.txt
export MSMS_BIN=$GLINT_ROOT/external/msms
export HHBLITS_BIN=$GLINT_ROOT/external/hhblits-bin

# replace with your own hh-suite database
export HHDB=$GLINT_ROOT/scratch/uniclust30_2016_09/uniclust30_2016_09

if [ ! -f ${HHDB}_hhm.ffindex ]; then
    echo "ERROR: invalid or damaged sequence database: $HHDB"
    exit 1
fi
file_path="../data_collection/cv_splits/BioGRID/test1.txt"
pdb_path="../dmasif/surface_data/raw/01-af_pdbs/"
output_path="examples/alphafold/"

# Check if the file exists
if [ -f "$file_path" ]; then
    # Open the file for reading
    exec 3< "$file_path"
    
    # Read each line of the file
    while IFS= read -r line <&3; do
        # Print or process each line as needed
        echo "$line"

        # Use IFS and read to split at tab
        IFS=$'\t' read -r prot1 prot2 <<< "$line"

        # Construct full .pdb paths
        receptor="${pdb_path}${prot1}.pdb"
        ligand="${pdb_path}${prot2}.pdb"
        start_time=$(date +%s.%N)
        time $GLINT_ROOT/scripts/build_hetero.sh $receptor $ligand $output_path
        # Capture the end time
        end_time=$(date +%s.%N)
        runtime=$(echo "$end_time - $start_time" | bc)
        echo $receptor
        echo $ligand
        #echo "Runtime $i: $runtime seconds" >> "test_biogrid.txt"

    done
    
    # Close the file descriptor
    exec 3<&-
else
    echo "File not found: $file_path"
fi

#bash $GLINT_ROOT/scripts/build_hetero.sh 