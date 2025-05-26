import pymol
from pymol import cmd,stored
import os
import sys
import re
import numpy as np
from dx2feature import *
from getResiLabel import *
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
import subprocess
import argparse
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='PInet pre-processing')
parser.add_argument('--pdb', type=str, default='1a02_F_J', help='PDB ID')
parser.add_argument('--pdb_list', type=str, default='pdb_list.txt', help='path to list with pdb pairs')
parser.add_argument('--database', type=str, default='exon/', help='Database name')
parser.add_argument('--train', type=bool, default=True, help='create labels for training')
parser.add_argument('--apbs', type=bool, default=True, help='create apbs features')
parser.add_argument('--pdbpath', type=str, default='exon/pdb/', help='path to pdb files')

args = parser.parse_args()
pdb_folder = args.pdbpath
with open(args.pdb_list, 'r') as f:
    pdb_list = f.read().splitlines()

print(pdb_list)

for pdb in tqdm(pdb_list):
    start = time.time()
    #pdb = args.pdb
    pdbs=pdb.split('_')

    pdbfile_l=pdbs[0]+'_'+pdbs[1]+'.pdb'
    pdbfile_r=pdbs[0]+'_'+pdbs[2]+'.pdb'

    train_flag=args.train
    needapbs=args.apbs

    data_folder = args.database
    # create subfolders if they are not existing
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(data_folder + 'lf/'):
        os.makedirs(data_folder + 'lf/')
    if not os.path.exists(data_folder + 'rf/'):
        os.makedirs(data_folder + 'rf/')
    if not os.path.exists(data_folder + 'lf/points_label/'):
        os.makedirs(data_folder + 'lf/points_label/')
    if not os.path.exists(data_folder + 'rf/points_label/'):
        os.makedirs(data_folder + 'rf/points_label/')
    if not os.path.exists(data_folder + 'lf/points/'):
        os.makedirs(data_folder + 'lf/points/')
    if not os.path.exists(data_folder + 'rf/points/'):
        os.makedirs(data_folder + 'rf/points/')



    label_l_folder = data_folder + "lf/points_label/"
    label_r_folder = data_folder + "rf/points_label/"
    pts_l_folder = data_folder + "lf/points/"
    pts_r_folder = data_folder + "rf/points/"


    #if len(sys.argv)==4 and sys.argv[3]=='train':
    #    train_flag=True

    cmd.load(pdb_folder + pdbfile_l)
    cmd.set('surface_quality', '0')
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    cmd.save(pdbfile_l[0:4]+'-l.wrl')
    cmd.delete('all')

    cmd.load(pdb_folder + pdbfile_r)
    cmd.set('surface_quality', '0')
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    cmd.save(pdbfile_r[0:4]+'-r.wrl')
    cmd.delete('all')

    # wrl to pts

    holder = []
    normholder =[]
    cf=0
    nf=0
    with open(pdbfile_l[0:4]+'-l.wrl', "r") as vrml:
        for lines in vrml:
            if 'point [' in lines:
                cf=1
            if cf==1:
                a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
                if len(a) == 3:
                    holder.append(tuple(map(float, a)))
            if 'vector [' in lines:
                nf=1
            if nf==1:
                a = re.findall("[-0-9]{1}.[0-9]{4}", lines)
                if len(a) == 3:
                    normholder.append(tuple(map(float, a)))

    lcoord=np.array(holder)

    holder = []
    normholder =[]
    cf=0
    nf=0
    with open(pdbfile_r[0:4]+'-r.wrl', "r") as vrml:
        for lines in vrml:
            if 'point [' in lines:
                cf=1
            if cf==1:
                a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
                if len(a) == 3:
                    holder.append(tuple(map(float, a)))
            if 'vector [' in lines:
                nf=1
            if nf==1:
                a = re.findall("[-0-9]{1}.[0-9]{4}", lines)
                if len(a) == 3:
                    normholder.append(tuple(map(float, a)))

    rcoord=np.array(holder)

    lcoord=np.unique(lcoord,axis=0)
    rcoord=np.unique(rcoord,axis=0)

    if train_flag:

        tol=np.array([2,2,2])

        contact = (np.abs(np.asarray(lcoord[:, None]) - np.asarray(rcoord))<tol).all(2).astype(int)

        llabel=np.max(contact,axis=1)
        rlabel=np.max(contact,axis=0)

        # change all 1 to 2 and 0 to 1
        llabel[llabel==1]=2
        llabel[llabel==0]=1
        rlabel[rlabel==1]=2
        rlabel[rlabel==0]=1

        np.savetxt(label_l_folder + pdb + '.seg',llabel, fmt='%d')
        np.savetxt(label_r_folder + pdb + '.seg',rlabel, fmt='%d')

    subprocess.check_output('rm '+pdbfile_r[0:4]+'*', shell=True)

    # pdb 2 pqr
    # pdb2pqr='/path/to/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
    # apbsflag='--whitespace --ff=amber -v --apbs-input'
    # apbs='/path/to/apbs-pdb2pqr/bin/apbs'

    if needapbs:
        pdb2pqr='pdb2pqr30'
        apbsflag=' --whitespace --ff=AMBER --apbs-input'
        apbs='apbs'

        try:
            os.system(pdb2pqr+' '+pdb_folder + pdbfile_l+' '+pdbfile_l[0:4]+'-l.pqr' + apbsflag + ' ' + pdbfile_l[0:4]+'-l.in')
        except:
            print('error when pdb2pqr l: '+pdbfile_l[0:4])

        try:
            os.system(pdb2pqr+' '+pdb_folder + pdbfile_r+' '+pdbfile_r[0:4]+'-r.pqr' + apbsflag + ' ' + pdbfile_r[0:4]+'-r.in')
        except:
            print('error when pdb2pqr r: '+pdbfile_r[0:4])

        try:
            os.system(apbs+' '+pdbfile_l[0:4]+'-l.in')
        except:
            print('error when abps l: '+pdbfile_l[0:4])

        try:
            os.system(apbs+' '+pdbfile_r[0:4]+'-r.in')
        except:
            print('error when abps r: '+pdbfile_r[0:4])


    # add apbs feature

    centroid_l, labelsl = gethydro(pdb_folder+pdbfile_l)
    centroid_r, labelsr = gethydro(pdb_folder+pdbfile_r)


    centroid_l = np.array(centroid_l)
    centroid_r = np.array(centroid_r)

    hlabell = np.transpose(np.asarray(labelsl[0]))
    hlabelr = np.transpose(np.asarray(labelsr[0]))

    clfl = neighbors.KNeighborsClassifier(3)
    clfr = neighbors.KNeighborsClassifier(3)

    clfl.fit(centroid_l,hlabell*10)
    clfr.fit(centroid_r,hlabelr*10)

    distl,indl=clfl.kneighbors(lcoord)
    distr,indr= clfr.kneighbors(rcoord)

    apbsl=open(pdbfile_l[0:4]+'-l.pqr.dx','r')
    apbsr = open(pdbfile_r[0:4] + '-r.pqr.dx','r')

    gl, orl, dl, vl = parsefile(apbsl)
    gr, orr, dr, vr = parsefile(apbsr)


    avl = findvalue(lcoord, gl, orl, dl, vl)
    avr = findvalue(rcoord, gr, orr, dr, vr)



    lpred=np.sum(hlabell[indl]*distl,1)/np.sum(distl,1)/10.0
    rpred = np.sum(hlabelr[indr] * distr, 1) / np.sum(distr, 1)/10.0


    np.savetxt(pts_l_folder + pdbfile_l[:-4]+ '.pts',np.concatenate((lcoord,np.expand_dims(avl,1),np.expand_dims(lpred,1)),axis=1))
    np.savetxt(pts_r_folder + pdbfile_r[:-4]+ '.pts',np.concatenate((rcoord, np.expand_dims(avr,1),np.expand_dims(rpred,1)),axis=1))

    subprocess.check_output('rm '+pdbfile_r[0:4]+'*', shell=True)
    end = time.time()
    print(end - start)
