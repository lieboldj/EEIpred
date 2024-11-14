import os
from tqdm import tqdm

if __name__ == '__main__':
    # open files.txt and read each line
    # files.txt contains the list of pdb pairs to be converted
    #with open('exon/EPPIC_EEIN_proc.txt') as f:
    #    pdb_pairs = f.read().splitlines()
    #
    #for pdb_pair in tqdm(pdb_pairs):
#
    #    if os.path.exists('exon/lf/points_label/'+pdb_pair+'.seg') and \
    #        os.path.exists('exon/rf/points_label/'+pdb_pair+'.seg'):
    #        continue
#
#
    #    if pdb_pair[0] == '#':
    #        print('skipping: ' + pdb_pair)
    #        continue
    #    try:
    #        os.system("python PreProcessLifeSavor.py "+pdb_pair)
    #    except:
    #        print("error in "+pdb_pair)
    #        continue
    
    with open('exon/PISA_EEIN_0.5_proc.txt') as f:
        pdb_pairs = f.read().splitlines()
    
    for pdb_pair in tqdm(pdb_pairs):
        if os.path.exists('exon/lf/points_label/'+pdb_pair+'.seg') and \
            os.path.exists('exon/rf/points_label/'+pdb_pair+'.seg'):
            continue
        
        if pdb_pair[0] == '#':
            print('skipping: ' + pdb_pair)
            continue
        try:
            os.system("python PreProcessLifeSavor.py "+pdb_pair)
        except:
            print("error in "+pdb_pair)
            continue
#
    