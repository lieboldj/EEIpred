import sys
import os
#import pdb2pqr
#import apbs

folderp='/home/liebold/Nextcloud/Projects/pinet/data/input_pdb/'

foldera='output_files/'

if not os.path.exists(foldera):
    os.mkdir(foldera)
#pdb2pqr='/path/to/pdb2pqr-linux-bin64-2.1.0/pdb2pqr'
#pdb2pqr = "/home/liebold/Nextcloud/Projects/pinet/pdb2pqr"
pdb2pqr = "/home/liebold/anaconda3/envs/pinet/bin/pdb2pqr30"
pdb2pqr = "pdb2pqr30"
apbsflag='--whitespace --ff=AMBER '

#apbs='/path/to/apbs-pdb2pqr/bin/apbs'

apbs='/home/liebold/anaconda3/envs/pinet/bin/apbs'
apbs = "apbs"

files=os.listdir(folderp)
os.chdir(foldera)
for f in files:
    if f[-5]=='l':
        continue
    print(pdb2pqr+' '+apbsflag+' '+folderp+f+' '+foldera+f[0:4]+'-l.pqr')
    try:
        os.system(pdb2pqr+' '+apbsflag+' '+folderp+f+' '+foldera+f[0:6]+".pqr")
        os.system(pdb2pqr+' '+apbsflag+' '+folderp+f[0:4]+'-l.pdb'+' '+foldera+f[0:4]+'-l.pqr')
    except:
        print ('pqr: '+f)

    try:
        os.system(pdb2pqr+' '+apbsflag+' '+folderp+f[0:4]+'-r.pdb'+' '+foldera+f[0:4]+'-r.pqr')
    except:
        print ('pqr: '+f)

    #try:
    #    # os.chdir(foldera)
    #    os.system(apbs+' '+foldera+f[0:4]+'-l.in')
    #except:
    #    print ('abps: '+f)
#
    #try:
    #    # os.chdir(foldera)
    #    os.system(apbs+' '+foldera+f[0:4]+'-r.in')
    #except:
    #    print ('abps: '+f)
