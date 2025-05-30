import os
import json
import numpy as np

import torch
import torch.nn.functional as F
import pickle

from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import sys
from scipy.spatial import distance





class ProteinDataset(data.Dataset):
    def __init__(self, file,shuffle=False,aug=True,centroid=True,subsample=False,mul=30,folder='dbdapbsfix'):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
        self.filelist = json.load(open(file, 'r'))
#         with open(file, 'wb') as inFH:
#             self.filelist=pickle.load(inFH)
        self.mul=mul
        self.folder=folder


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
        plf=self.folder+'/lf/points/'+self.filelist[cindex][-6:]+'.pts'
        prf=self.folder+'/rf/points/'+self.filelist[cindex][-6:-1]+'r.pts'
        
        llf=self.folder+'/lf/points_label/'+self.filelist[cindex][-6:]+'.seg'
        lrf=self.folder+'/rf/points_label/'+self.filelist[cindex][-6:-1]+'r.seg'
        
        dplf=np.loadtxt(plf)
        dprf=np.loadtxt(prf)
        dllf=np.loadtxt(llf)
        dlrf=np.loadtxt(lrf)
        
#         if dplf.shape[0]>dprf.shape[0]:
#             t=dprf
#             dprf=dplf
#             dplf=t
            
#             t=dlrf
#             dlrf=dllf
#             dllf=t
        
        if self.subsample:
            memlim=20000
            if dplf.shape[0]+dprf.shape[0]>memlim:
                pickl=dplf.shape[0]*memlim/(dplf.shape[0]+dprf.shape[0])
                pickr = dprf.shape[0] * memlim / (dplf.shape[0] + dprf.shape[0])
                ls=np.random.choice(dplf.shape[0], int(pickl), replace=False)
                rs=np.random.choice(dprf.shape[0], int(pickr), replace=False)
                dprf=dprf[rs,:]
                dlrf=dlrf[rs]
                dplf=dplf[ls,:]
                dllf=dllf[ls]
        
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,5))
#             dprf=dprf.reshape((dprf.shape[0],-1,5))
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            
#             rl=np.max(np.sum(dplf[:,[0,1,2]]**2,0)**0.5)
#             rr=np.max(np.sum(dprf[:,[0,1,2]]**2,0)**0.5)
#             rall=max(rl,rr)
            
#             dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/rall
#             dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/rall
            
            
            
#             distl = np.max(np.sqrt(dplf[:,[3,4]] ** 2),0)
#             distr = np.max(np.sqrt(dprf[:,[3,4]] ** 2),0)
#             distall=np.maximum(distl,distr)
            
#             dplf[:,[3,4]] = dplf[:,[3,4]] / distall #scale
            dplf[:,[3,4]] = dplf[:,[3,4]]
#             distr = np.max(np.sqrt(dprf[:,[3,4]] ** 2),0)
#             dprf[:,[3,4]] = dprf[:,[3,4]] / distall #scale
            dprf[:,[3,4]] = dprf[:,[3,4]] 
#             distr = np.max(np.sqrt(np.sum(dprf[:,[0,1,2]] ** 2, axis = 1)),0)
#             dprf[:,[0,1,2]] = dprf[:,[0,1,2]] / distr #scale
            
        
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)


class ProteinDataset_resi(data.Dataset):
    def __init__(self, file,folder='dbdresi300/',indim=5,numofpoint=600,aug=False,centroid=False,shuffle=False,geo=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.data_augmentation=aug
        self.filelist = json.load(open(file, 'r'))
        self.folder=folder
        self.indim=indim
        self.geo=geo
        self.nop=numofpoint


    def __getitem__(self, index):
#         plf='dbdapbscon/lf/points/'+self.filelist[index][-6:]+'.pts'
#         prf='dbdapbscon/rf/points/'+self.filelist[index][-6:-1]+'r.pts'
        
#         llf='dbdapbscon/lf/points_label/'+self.filelist[index][-6:]+'.seg'
#         lrf='dbdapbscon/rf/points_label/'+self.filelist[index][-6:-1]+'r.seg'
        cindex=index%len(self.filelist)
        
#         with open('dbdresi_ball/'+self.filelist[cindex][-6:-2].upper()+'.pickle', 'rb') as inFH:
#             dcl,dcr,dplf,dprf,dllf,dlrf = pickle.load(inFH)
        with open(self.folder+self.filelist[cindex][-6:-2].upper()+'.pickle', 'rb') as inFH:
            dcl,dcr,dplf,dprf,dllf,dlrf,dmat = pickle.load(inFH)
            
#         dmat=distance.cdist(dcl,dcr)
#         (145, 600, 5)
#         dplf=dplf[:,0:self.nop,0:self.indim]
#         dprf=dprf[:,0:self.nop,0:self.indim]
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,self.indim))
#             dprf=dprf.reshape((dprf.shape[0],-1,self.indim))
            
#             dplf[:,:,[0,1,2]] = dplf[:,:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,:,[0,1,2]], axis = 0), 0) # center
#             dprf[:,:,[0,1,2]] = dprf[:,:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,:,[0,1,2]], axis = 0), 0) # center
            
            dplf[:,:,[0,1,2]] = dplf[:,:,[0,1,2]] - np.expand_dims(np.mean(dcl, axis = 0), 0) # center
            dprf[:,:,[0,1,2]] = dprf[:,:,[0,1,2]] - np.expand_dims(np.mean(dcr, axis = 0), 0) # center
        
            dcl = dcl - np.expand_dims(np.mean(dcl, axis = 0), 0) # center
            dcr = dcr - np.expand_dims(np.mean(dcr, axis = 0), 0) # center
        
        if self.data_augmentation:
            
#             theta = np.random.uniform(0,np.pi*2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
#             dcl[:,[0,2]] = dcl[:,[0,2]].dot(rotation_matrix) # random rotation
#             dcl += np.random.normal(0, 0.02, size=dcl.shape) # random jitter
# #             dcr[:,[0,2]] = dcr[:,[0,2]].dot(rotation_matrix) # random rotation
# #             dcr += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

#             dplf[:,:,[0,2]] = dplf[:,:,[0,2]].dot(rotation_matrix) # random rotation
#             dplf[:,:,[0,2]] += np.random.normal(0, 0.02, size=dplf[:,:,[0,2]].shape) # random jitter
        
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+6 for i in roi]
            eroi=[i+9 for i in roi]
            dcl[:,roi ]= dcl[:,roi].dot(rotation_matrix) # random rotation
            dcl += np.random.normal(0, 0.02, size=dcl.shape) # random jitter
            dplf[:,:,roi] = dplf[:,:,roi].dot(rotation_matrix) # random rotation
            dplf[:,:,groi] = dplf[:,:,groi].dot(rotation_matrix) # random rotation
            dplf[:,:,eroi] = dplf[:,:,eroi].dot(rotation_matrix) # random rotation
            dplf[:,:,:] += np.random.normal(0, 0.02, size=dplf[:,:,:].shape) # random jitter
        
#         dplf=np.loadtxt(plf)
#         dprf=np.loadtxt(prf)
#         dllf=np.loadtxt(llf)
#         dlrf=np.loadtxt(lrf)

#         dplf=np.transpose(dplf,(dplf.shape[0],self.indim,-1))
#         dprf=np.transpose(dprf,(dprf.shape[0],self.indim,-1))
#         dplf=dplf.transpose(1,2)
#         dprf=dprf.transpose(1,2)

        dplf=dplf[:,0:self.nop,0:self.indim]
        dprf=dprf[:,0:self.nop,0:self.indim]

        dplf=np.transpose(dplf,(0,2,1))
        dprf=np.transpose(dprf,(0,2,1))
        
        if self.geo:
            dplf=dplf[:,0:3,:]
            dprf=dprf[:,0:3,:]

#         dmat=distance.cdist(dcl,dcr)
        
        
        ml=np.zeros(dllf.shape)
        mr=np.zeros(dlrf.shape)

        
#         return torch.from_numpy(dcl).float(),torch.from_numpy(dcr).float(),torch.from_numpy(dplf).transpose(0,1).float(),torch.from_numpy(dprf).transpose(0,1).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ml).float(),torch.from_numpy(mr).float()
        return torch.from_numpy(dcl).float(),torch.from_numpy(dcr).float(),torch.from_numpy(dplf).transpose(0,1).float(),torch.from_numpy(dprf).transpose(0,1).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ml),torch.from_numpy(mr),torch.from_numpy(dmat)

    def __len__(self):
        if self.data_augmentation:
            return len(self.filelist)*self.data_augmentation
        else:
            return len(self.filelist)


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

    def collater(self, samples):
#         resil_l = [dplf.shape[1] for dcl,dcr,dplf,dprf,dllf,dlrf in samples]+[dprf.shape[1] for dcl,dcr,dplf,dprf,dllf,dlrf in samples]
        resil_l = [dplf.shape[1] for dcl,dcr,dplf,dprf,dllf,dlrf,_,_,_ in samples]
        max_resi_l = max(resil_l)
        
        resil_r = [dprf.shape[1] for dcl,dcr,dplf,dprf,dllf,dlrf,_,_,_ in samples]
        max_resi_r = max(resil_r)
        
#         pointl = [dplf.shape[2] for dcl,dcr,dplf,dprf,dllf,dlrf in samples]+[dprf.shape[2] for dcl,dcr,dplf,dprf,dllf,dlrf in samples]
        pointl_l = [dplf.shape[2] for dcl,dcr,dplf,dprf,dllf,dlrf,_,_,_ in samples]
        max_point_l = max(pointl_l)
        
        pointl_r = [dprf.shape[2] for dcl,dcr,dplf,dprf,dllf,dlrf,_,_,_ in samples]
        max_point_r = max(pointl_r)

        dcl_p = []
        dcr_p = []
        dplf_p = []
        dprf_p = []
        dllf_p = []
        dlrf_p = []
        ml_p=[]
        mr_p=[]
        dmat_p=[]

#         for (features, locations), n in zip(samples, num_objects):
#             features_padded = F.pad(features, pad=[0, 0, 0, max_objects-n], mode='constant', value=-1.0)
# #             locations_padded = F.pad(locations, pad=[0, 0, 0, max_objects-n], mode='constant', value=0.0)
#             feature_samples_padded.append(features_padded)
# #             location_samples_padded.append(locations_padded)

# #         return default_collate(feature_samples_padded), default_collate(location_samples_padded)
#         return default_collate(feature_samples_padded)
# #         return self.s2s_collater.collate(samples)

        for dcl,dcr,dplf,dprf,dllf,dlrf,ml,mr,dmat in samples:
            curr_resi_len_l=dplf.shape[1]
            curr_resi_len_r=dprf.shape[1]
            
            curr_point_len_l=dplf.shape[2]
            curr_point_len_r=dprf.shape[2]
        
        
            dclp = F.pad(dcl, pad=[0,0,0, max_resi_l-curr_resi_len_l], mode='constant', value=0)
            dcl_p.append(dclp)
            
            dcrp = F.pad(dcr, pad=[0,0,0, max_resi_r-curr_resi_len_r], mode='constant', value=0)
            dcr_p.append(dcrp)
            
            dplfp = F.pad(dplf, pad=[0,max_point_l-curr_point_len_l,0, max_resi_l-curr_resi_len_l], mode='constant', value=0)
            dplf_p.append(dplfp)
            
            dprfp = F.pad(dprf, pad=[0,max_point_r-curr_point_len_r,0, max_resi_r-curr_resi_len_r], mode='constant', value=0)
            dprf_p.append(dprfp)
            
            dllfp = F.pad(dllf, pad=[0, max_resi_l-curr_resi_len_l], mode='constant', value=0)
            dllf_p.append(dllfp)
            
            dlrfp = F.pad(dlrf, pad=[0, max_resi_r-curr_resi_len_r], mode='constant', value=0)
            dlrf_p.append(dlrfp)
            
            mlp = F.pad(ml, pad=[0, max_resi_l-curr_resi_len_l], mode='constant', value=1)
            ml_p.append(mlp)
            
            mrp = F.pad(mr, pad=[0, max_resi_r-curr_resi_len_r], mode='constant', value=1)
            mr_p.append(mrp)
            
#             dmatp = F.pad(dmat, pad=[0, max_resi_l-curr_resi_len_l,0, max_resi_r-curr_resi_len_r], mode='constant', value=0)
            dmatp = F.pad(dmat, pad=[0, max_resi_r-curr_resi_len_r,0, max_resi_l-curr_resi_len_l], mode='constant', value=0)

            dmat_p.append(dmatp)

        return default_collate(dcl_p),default_collate(dcr_p),default_collate(dplf_p),default_collate(dprf_p),default_collate(dllf_p),default_collate(dlrf_p),default_collate(ml_p),default_collate(mr_p),default_collate(dmat_p)

    
class ProteinDatasetFrag(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,subsample=False,mul=30,folder='dbdapbsfix'):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
#         self.filelist = json.load(open(file, 'r'))
        self.mul=mul
        self.folder=folder


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
        plf=self.folder+'/pts2/'+self.filelist[cindex][-6:-1]+'l.pts'
        prf=self.folder+'/pts2/'+self.filelist[cindex][-6:-1]+'r.pts'
        
        llf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.seg'
        lrf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.seg'
        
        dplf=np.loadtxt(plf)
        dprf=np.loadtxt(prf)
        dllf=np.loadtxt(llf)
        dlrf=np.loadtxt(lrf)
        
        if self.subsample:
            memlim=20000
            if dplf.shape[0]+dprf.shape[0]>memlim:
                pickl=dplf.shape[0]*memlim/(dplf.shape[0]+dprf.shape[0])
                pickr = dprf.shape[0] * memlim / (dplf.shape[0] + dprf.shape[0])
                ls=np.random.choice(dplf.shape[0], int(pickl), replace=False)
                rs=np.random.choice(dprf.shape[0], int(pickr), replace=False)
                dprf=dprf[rs,:]
                dlrf=dlrf[rs]
                dplf=dplf[ls,:]
                dllf=dllf[ls]
        
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,5))
#             dprf=dprf.reshape((dprf.shape[0],-1,5))
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            
#             rl=np.max(np.sum(dplf[:,[0,1,2]]**2,0)**0.5)
#             rr=np.max(np.sum(dprf[:,[0,1,2]]**2,0)**0.5)
#             rall=max(rl,rr)
            
#             dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/rall
#             dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/rall
            
            
            
#             distl = np.max(np.sqrt(dplf[:,[3,4]] ** 2),0)
#             distr = np.max(np.sqrt(dprf[:,[3,4]] ** 2),0)
#             distall=np.maximum(distl,distr)
            
#             dplf[:,[3,4]] = dplf[:,[3,4]] / distall #scale
            dplf[:,[3,4]] = dplf[:,[3,4]]
#             distr = np.max(np.sqrt(dprf[:,[3,4]] ** 2),0)
#             dprf[:,[3,4]] = dprf[:,[3,4]] / distall #scale
            dprf[:,[3,4]] = dprf[:,[3,4]] 
#             distr = np.max(np.sqrt(np.sum(dprf[:,[0,1,2]] ** 2, axis = 1)),0)
#             dprf[:,[0,1,2]] = dprf[:,[0,1,2]] / distr #scale
            
        
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,eroi] = dplf[:,eroi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,eroi] = dprf[:,eroi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)


class ProteinDatasetSam(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,normsize=0,subsample=True,mul=30,exppower=1,folder='dbdapbsfix',nop=2048,dcut=10,powerd=1,hf=0,kl=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
        self.filelist = json.load(open(file, 'r'))
        #self.filelist = list()
        #self.filelist.append("shape_data/lf/2YVJ-l")
        self.mul=mul
        self.folder=folder
        self.nop=nop
        self.pd=powerd
        self.dcut=dcut
        self.half=hf
        self.expp=exppower
        self.normsize=normsize
        self.maxradius=76
        self.pkl=kl
        


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
        plf=self.folder+'/lf/points/'+self.filelist[cindex][-6:-1]+'l.pts'
        prf=self.folder+'/rf/points/'+self.filelist[cindex][-6:-1]+'r.pts'
        llf=self.folder+'/lf/points_label/'+self.filelist[cindex][-6:]+'.seg'
        lrf=self.folder+'/rf/points_label/'+self.filelist[cindex][-6:-1]+'r.seg'

        #if self.pkl:
        #    dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.ssseg'
        #    drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.ssseg'
        #else:
        #    dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.sseg'
        #    drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.sseg'
        
        tdplf=np.loadtxt(plf)
        tdprf=np.loadtxt(prf)

        if tdplf.shape[0]>tdprf.shape[0]:
            
            dplf=tdplf
            dprf=tdprf
        
            dllf=np.loadtxt(llf)
            dlrf=np.loadtxt(lrf)
            
            #ddlf=np.loadtxt(dlf)
            #ddrf=np.loadtxt(drf)
        else:
            dplf=tdprf
            dprf=tdplf
        
            dllf=np.loadtxt(lrf)
            dlrf=np.loadtxt(llf)
            
            #ddlf=np.loadtxt(drf)
            #ddrf=np.loadtxt(dlf)
            
        if self.pd==None:
            pass
        #elif self.pd[0:3]=='exp':
        #    ddlf=np.exp(-(np.minimum(ddlf,self.dcut)/float(self.pd[3:]))**self.expp)
        #    ddrf=np.exp(-(np.minimum(ddrf,self.dcut)/float(self.pd[3:]))**self.expp)
        #else:
        #    ddlf=(1-np.minimum(ddlf,self.dcut)/self.dcut)**float(self.pd)
        #    ddrf=(1-np.minimum(ddrf,self.dcut)/self.dcut)**float(self.pd)
            
        
        if self.subsample:
           
            if dplf.shape[0]>self.nop:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=False)
            else:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=True)
                
            if self.half:
                sn=int(self.nop/2)
            else:
                sn=int(self.nop)
            if dprf.shape[0]>self.nop:
                
                rs=np.random.choice(dprf.shape[0], sn, replace=False)
            else:
                rs=np.random.choice(dprf.shape[0], sn, replace=True)
            dprf=dprf[rs,:]
            dlrf=dlrf[rs]
            #ddrf=ddrf[rs]
            dplf=dplf[ls,:]
            dllf=dllf[ls]
            #ddlf=ddlf[ls]
        
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,5))
#             dprf=dprf.reshape((dprf.shape[0],-1,5))
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            if self.normsize:
                dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/self.maxradius
                dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/self.maxradius
            
            dplf[:,3]=np.maximum(np.minimum(dplf[:,3],50),-50)/50
            dprf[:,3]=np.maximum(np.minimum(dprf[:,3],50),-50)/50
            
            dplf[:,4]=dplf[:,4]/4.5
            dprf[:,4]=dprf[:,4]/4.5

            
            
            #dprf[:,[5,6,7]] = dplf[:,[5,6,7]] / np.expand_dims(np.sum(dplf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            #dprf[:,[5,6,7]] = dprf[:,[5,6,7]] / np.expand_dims(np.sum(dprf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            
            
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            print(groi, eroi, roi, dplf.shape, dprf.shape)
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,eroi] = dplf[:,eroi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,eroi] = dprf[:,eroi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float()#,torch.from_numpy(ddlf).float(),torch.from_numpy(ddrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)

class ProteinDatasetSamMasifPP(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,normsize=0,subsample=True,mul=30,exppower=1,folder='masif',nop=2048,dcut=10,powerd=1,hf=0,kl=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
#         self.filelist = json.load(open(file, 'r'))
        with open(file, "rb") as input_file:
            self.filelist = list(pickle.load(input_file).keys())
        self.mul=mul
        self.folder=folder
        self.nop=nop
        self.pd=powerd
        self.dcut=dcut
        self.half=hf
        self.expp=exppower
        self.normsize=normsize
        self.maxradius=100
        self.pkl=kl
        


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
#         plf=self.folder+'/pts2/'+self.filelist[cindex][-6:-1]+'l.pts'
#         prf=self.folder+'/pts2/'+self.filelist[cindex][-6:-1]+'r.pts'
        
#         llf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.seg'
#         lrf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.seg'
        
#         if self.pkl:
#             dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.ssseg'
#             drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.ssseg'
#         else:
#             dlf=self.folder+'/seg/'+self.filelist[cindex][-6:]+'.sseg'
#             drf=self.folder+'/seg/'+self.filelist[cindex][-6:-1]+'r.sseg'

        plf=self.folder+'/pts3/'+self.filelist[cindex]+'-l.pts'
        prf=self.folder+'/pts3/'+self.filelist[cindex]+'-r.pts'
        
        llf=self.folder+'/pts/'+self.filelist[cindex]+'-l.seg'
        lrf=self.folder+'/pts/'+self.filelist[cindex]+'-r.seg'
        
        dlf=self.folder+'/pts/'+self.filelist[cindex]+'-l.ssseg'
        drf=self.folder+'/pts/'+self.filelist[cindex]+'-r.ssseg'
        
        tdplf=np.loadtxt(plf)
        tdprf=np.loadtxt(prf)
        
        if tdplf.shape[0]>tdprf.shape[0]:
            
            dplf=tdplf
            dprf=tdprf
        
            dllf=np.loadtxt(llf)
            dlrf=np.loadtxt(lrf)
            
            ddlf=np.loadtxt(dlf)
            ddrf=np.loadtxt(drf)
        else:
            dplf=tdprf
            dprf=tdplf
        
            dllf=np.loadtxt(lrf)
            dlrf=np.loadtxt(llf)
            
            ddlf=np.loadtxt(drf)
            ddrf=np.loadtxt(dlf)
            
        if self.pd==None:
            pass
        elif self.pd[0:3]=='exp':
            ddlf=np.exp(-(np.minimum(ddlf,self.dcut)/float(self.pd[3:]))**self.expp)
            ddrf=np.exp(-(np.minimum(ddrf,self.dcut)/float(self.pd[3:]))**self.expp)
        else:
            ddlf=(1-np.minimum(ddlf,self.dcut)/self.dcut)**float(self.pd)
            ddrf=(1-np.minimum(ddrf,self.dcut)/self.dcut)**float(self.pd)
            
        
        if self.subsample:
           
            if dplf.shape[0]>self.nop:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=False)
            else:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=True)
                
            if self.half:
                sn=int(self.nop/2)
            else:
                sn=int(self.nop)
            if dprf.shape[0]>self.nop:
                
                rs=np.random.choice(dprf.shape[0], sn, replace=False)
            else:
                rs=np.random.choice(dprf.shape[0], sn, replace=True)
            dprf=dprf[rs,:]
            dlrf=dlrf[rs]
            ddrf=ddrf[rs]
            dplf=dplf[ls,:]
            dllf=dllf[ls]
            ddlf=ddlf[ls]
        
        if self.centroid:
#             dplf=dplf.reshape((dplf.shape[0],-1,5))
#             dprf=dprf.reshape((dprf.shape[0],-1,5))
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            if self.normsize:
                dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/self.maxradius
                dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/self.maxradius
            
            dplf[:,3]=np.maximum(np.minimum(dplf[:,3],50),-50)/50
            dprf[:,3]=np.maximum(np.minimum(dprf[:,3],50),-50)/50
            
            dplf[:,4]=dplf[:,4]/4.5
            dprf[:,4]=dprf[:,4]/4.5
            
            dplf[:,[5,6,7]] = dplf[:,[5,6,7]] / np.expand_dims(np.sum(dplf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            dprf[:,[5,6,7]] = dprf[:,[5,6,7]] / np.expand_dims(np.sum(dprf[:,[5,6,7]]**2, axis = 1)**0.5,1)
            
            
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,eroi] = dplf[:,eroi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,eroi] = dprf[:,eroi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ddlf).float(),torch.from_numpy(ddrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)

class ProteinDatasetSamMasif(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,subsample=True,mul=30,folder='masif',nop=2048,normsize=0):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
        with open(file, "rb") as input_file:
            self.filelist = list(pickle.load(input_file).keys())
        self.mul=mul
        self.folder=folder
        self.nop=nop
        self.normsize=normsize


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        
        plf=self.folder+'/pts/'+self.filelist[cindex]+'-l.pts'
        prf=self.folder+'/pts/'+self.filelist[cindex]+'-r.pts'
        
        llf=self.folder+'/pts/'+self.filelist[cindex]+'-l.seg'
        lrf=self.folder+'/pts/'+self.filelist[cindex]+'-r.seg'
        
        dlf=self.folder+'/pts/'+self.filelist[cindex]+'-l.ssseg'
        drf=self.folder+'/pts/'+self.filelist[cindex]+'-r.ssseg'
        
        
        dplf=np.loadtxt(plf)
        dprf=np.loadtxt(prf)
        dllf=np.loadtxt(llf)
        dlrf=np.loadtxt(lrf)
        ddlf=np.loadtxt(dlf)
        ddrf=np.loadtxt(drf)
        
        ddlf=np.exp(-np.minimum(ddlf,20)/2)
        ddrf=np.exp(-np.minimum(ddrf,20)/2)
        
        if self.subsample:
           
            if dplf.shape[0]>self.nop:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=False)
            else:
                ls=np.random.choice(dplf.shape[0], self.nop, replace=True)
            if dprf.shape[0]>self.nop:
                
                rs=np.random.choice(dprf.shape[0], self.nop, replace=False)
            else:
                rs=np.random.choice(dprf.shape[0], self.nop, replace=True)
            dprf=dprf[rs,:]
            dlrf=dlrf[rs]
            ddrf=ddrf[rs]
            dplf=dplf[ls,:]
            dllf=dllf[ls]
            ddlf=ddlf[ls]
        
        if self.centroid:
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center
            
            if self.normsize:
#                 rl=np.max(np.sum(dplf[:,[0,1,2]]**2,0)**0.5)
#                 rr=np.max(np.sum(dprf[:,[0,1,2]]**2,0)**0.5)
#                 rall=max(rl,rr)

                dplf[:,[0,1,2]] = dplf[:,[0,1,2]]/100
                dprf[:,[0,1,2]] = dprf[:,[0,1,2]]/100
            
        
        if self.data_augmentation:
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+6 for i in roi]
            dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
            dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
            dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+6 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float(),torch.from_numpy(ddlf).float(),torch.from_numpy(ddrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices




class ProteinDatasetFragPre(data.Dataset):
    def __init__(self, file,shuffle=True,aug=True,centroid=True,subsample=False,mul=30,folder='dbdapbsfix'):
        self.shuffle=shuffle
        self.centroid=centroid
        self.file = file
        self.subsample=subsample
        self.data_augmentation=aug
#         self.filelist = json.load(open(file, 'r'))
        with open(file, 'rb') as inFH:
            self.filelist=pickle.load(inFH)
        self.mul=mul
        self.folder=folder


    def __getitem__(self, index):
        
        cindex=index%len(self.filelist)
        pdb,lind,rind=self.filelist[cindex]
        
        plf=self.folder+'/ptsfrag/'+pdb+str(lind)+'-l.pts'
        prf=self.folder+'/ptsfrag/'+pdb+str(rind)+'-r.pts'
        
        dplf=np.loadtxt(plf)
        dprf=np.loadtxt(prf)
        
        lcoord=dplf[:,0:3]
        rcoord=dprf[:,0:3]
        
        tol=np.array([2,2,2])

        contact = (np.abs(np.asarray(lcoord[:, None]) - np.asarray(rcoord))<tol).all(2).astype(np.int)

        dllf=np.max(contact,axis=1)
        dlrf=np.max(contact,axis=0)
        
        if self.centroid:
            
            dplf[:,[0,1,2]] = dplf[:,[0,1,2]] - np.expand_dims(np.mean(dplf[:,[0,1,2]], axis = 0), 0) # center
            dprf[:,[0,1,2]] = dprf[:,[0,1,2]] - np.expand_dims(np.mean(dprf[:,[0,1,2]], axis = 0), 0) # center

            
        
        if self.data_augmentation:
            
#             theta = np.random.uniform(0,np.pi*2)
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
#             roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
#             groi=[i+5 for i in roi]
#             eroi=[i+8 for i in roi]
#             dplf[:,roi] = dplf[:,roi].dot(rotation_matrix) # random rotation
#             dplf[:,groi] = dplf[:,groi].dot(rotation_matrix) # random rotation
#             dplf[:,eroi] = dplf[:,eroi].dot(rotation_matrix) # random rotation
#             dplf[:,roi] += np.random.normal(0, 0.02, size=dplf[:,roi].shape) # random jitter
            
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            roi=sorted(np.random.choice([0,1,2],2,replace=False).tolist())
            groi=[i+5 for i in roi]
            eroi=[i+8 for i in roi]
            dprf[:,roi] = dprf[:,roi].dot(rotation_matrix) # random rotation
            dprf[:,groi] = dprf[:,groi].dot(rotation_matrix) # random rotation
            dprf[:,eroi] = dprf[:,eroi].dot(rotation_matrix) # random rotation
            dprf[:,roi] += np.random.normal(0, 0.02, size=dprf[:,roi].shape) # random jitter
        
        

        
        return torch.from_numpy(dplf).float(),torch.from_numpy(dprf).float(),torch.from_numpy(dllf).float(),torch.from_numpy(dlrf).float()

    def __len__(self):
        return len(self.filelist)*self.mul


    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            
        return indices

#     def collater(self, samples):
#         lenl = [pl.shape[0] for pl,pr,ll,lr in samples]
#         maxl = max(lenl)
        
#         lenr = [pr.shape[0] for pl,pr,ll,lr in samples]
#         maxr = max(lenr)

#         pl_pad = []
#         pr_pad = []
#         ll_pad = []
#         lr_pad = []

#         for (pl,pr,ll,lr), cl,cr in zip(samples, lenl,lenr):
#             features_padded = F.pad(features, pad=[0,0,0, max_objects-n], mode='constant', value=0)
            
#             llpad=
            
#             feature_samples_padded.append(features_padded)
#             label_samples_padded.append(label)

#         return default_collate(feature_samples_padded),default_collate(label_samples_padded)