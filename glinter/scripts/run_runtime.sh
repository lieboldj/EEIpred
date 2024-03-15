#! /usr/bin/bash

# read /cosybio/project/EEIP/EEIP/data_collection/cv_splits/RUNTIME/test1.txt
# read /cosybio/project/EEIP/EEIP/data_collection/cv_splits/RUNTIME/test_info1.txt
export GLINT_ROOT=/cosybio/project/EEIP/EEIP/glinter
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
file_path="/cosybio/project/EEIP/EEIP/data_collection/cv_splits/RUNTIME/test1.txt"
pdb_path="/cosybio/project/EEIP/EEIP/glinter/examples/PDB/"

# Check if the file exists
if [ -f "$file_path" ]; then
    # Open the file for reading
    exec 3< "$file_path"
    
    # Read each line of the file
    while IFS= read -r line <&3; do
        # Print or process each line as needed
        # split line at _ and get the first element + second and also first element + third
        receptor=$pdb_path$(echo $line | cut -d'_' -f1)_$(echo $line | cut -d'_' -f2).pdb
        ligand=$pdb_path$(echo $line | cut -d'_' -f1)_$(echo $line | cut -d'_' -f3).pdb
        # Capture the start time
        start_time=$(date +%s.%N)
        time $GLINT_ROOT/scripts/build_hetero.sh $receptor $ligand
        # Capture the end time
        end_time=$(date +%s.%N)
        runtime=$(echo "$end_time - $start_time" | bc)
        echo "Runtime $i: $runtime seconds" >> "pre_GLINTER.txt"

    done
    
    # Close the file descriptor
    exec 3<&-
else
    echo "File not found: $file_path"
fi

#bash $GLINT_ROOT/scripts/build_hetero.sh 