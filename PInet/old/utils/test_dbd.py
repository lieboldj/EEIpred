from __future__ import print_function
import sys
sys.path.append(".")
sys.path.append("../data/")
import argparse
import os
import random
import torch
import torch.nn.parameter
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pinet.dataset import ShapeNetDataset3aug
from pinet.model import PointNetDenseCls12, feature_transform_regularizer

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score,recall_score,roc_curve,auc, roc_auc_score
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def gk(x):
    cen=torch.nn.Parameter(torch.tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]).float(), requires_grad=False).cuda()
    xmat=x.float().cuda()-cen
    sigma=200
    y = torch.sum(torch.sigmoid(sigma * (xmat + 0.1 / 2)) - torch.sigmoid(sigma * (xmat - 0.1 / 2)),dim=0)
    y = y / torch.sum(y)
    return y

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--npoints', type=int, default=20000, help='subsample points')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--class_choice', type=str, default='protein', help="class_choice")
parser.add_argument('--r', type=str, default='recept', help="recept_choice")
parser.add_argument('--l', type=str, default='ligand', help="ligand_choice")
parser.add_argument('--fold', type=str, default='', help="kfold")
parser.add_argument('--bs2', type=int, default=8, help="bs")
parser.add_argument('--drop', type=int, default=0, help="droprate")
parser.add_argument('--ft', type=int, default=0, help="ft")
# from 16
parser.add_argument('--indim', type=int, default=5, help="input dim")
parser.add_argument('--start', type=int, default=10, help="start epoch")
parser.add_argument('--fac', type=float, default=100, help="start epoch")
parser.add_argument('--lloss', type=int, default=1, help="start epoch")
parser.add_argument('--rs', type=int, default=0, help="start epoch")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--dataset_name', type=str, default='_contact', help="dataset name")
parser.add_argument('--cuda', type=str, default="0", help='choose cuda GPU number')

opt = parser.parse_args()
if opt.ft==1:
    opt.feature_transform=True
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

data_name = opt.dataset_name

opt.class_choice = opt.class_choice + data_name

# data loading
dataset_r = ShapeNetDataset3aug(
    root=opt.dataset,
    dataset_name = data_name,
    npoints=3000,
    classification=False,
    class_choice=[opt.r],
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
dataloader_r = torch.utils.data.DataLoader(
    dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

dataset_l = ShapeNetDataset3aug(
    root=opt.dataset,
    dataset_name = data_name,
    npoints=3000,
    classification=False,
    class_choice=[opt.l],
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
dataloader_l = torch.utils.data.DataLoader(
    dataset_l,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_r = ShapeNetDataset3aug(
    root=opt.dataset,
    dataset_name = data_name,
    npoints=3000,
    classification=False,
    class_choice=[opt.r],
    split='test',
    data_augmentation=False,
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
testdataloader_r = torch.utils.data.DataLoader(
    test_dataset_r,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset_l = ShapeNetDataset3aug(
    root=opt.dataset,
    dataset_name = data_name,
    npoints=3000,
    classification=False,
    class_choice=[opt.l],
    split='test',
    data_augmentation=False,
    indim=opt.indim,
    rs=opt.rs,
    fold=opt.fold)
testdataloader_l = torch.utils.data.DataLoader(
    test_dataset_l,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

num_classes = dataset_l.num_seg_classes

try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'
# model
classifier = PointNetDenseCls12(k=num_classes, feature_transform=opt.feature_transform,pdrop=1.0*opt.drop/10.0,id=opt.indim)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()


num_batch = len(dataset_r) / opt.batchSize


all = []
allp = []
allr = []
allauc = []
for j, (datar, datal) in tqdm(enumerate(zip(testdataloader_r, testdataloader_l), 0)):
    pointsr, targetr = datar
    pointsl, targetl = datal
    memlim = 90000
    if pointsl.size()[1] + pointsr.size()[1] > memlim:
        lr = pointsl.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
        rr = pointsr.size()[1] * memlim / (pointsl.size()[1] + pointsr.size()[1])
        ls = np.random.choice(pointsl.size()[1], int(lr), replace=False)
        rs = np.random.choice(pointsr.size()[1], int(rr), replace=False)
        pointsr = pointsr[:, rs, :]
        targetr = targetr[:, rs]
        pointsl = pointsl[:, ls, :]
        targetl = targetl[:, ls]
    pointsr = pointsr.transpose(2, 1)
    pointsl = pointsl.transpose(2, 1)
    pointsr, targetr = pointsr.cuda(), targetr.cuda()
    pointsl, targetl = pointsl.cuda(), targetl.cuda()
    classifier = classifier.eval()
    try:
        pred, _, _ = classifier(pointsr, pointsl)
    except:
        print("fail")
        continue
    pred = pred.view(-1, 1)
    target = torch.cat((targetr, targetl), 1)
    target = target.view(-1, 1) - 1
    pred_choice = torch.gt(torch.sigmoid(pred.data), 0.5).long()
    try:
        correct0 = pred_choice.eq(target.data).cpu().sum()
        correct1 = (pred_choice.eq(target.data).long() * target.data).cpu().sum()
    except:
        print("fail continue")
        continue
    blue = lambda x: '\033[94m' + x + '\033[0m'
    if j==0:
        print('[%d] %s  accuracy: %f' % (num_batch, blue('test'), correct0.item() / float(opt.batchSize * target.size()[0])))
    try:
        roc = roc_auc_score(target.data.cpu(), torch.sigmoid(pred.data).cpu())
    except:
        print("fail")
        continue
    all.append(correct0.item() / float(opt.batchSize * target.size()[0]))
    allp.append(precision_score(target.data.cpu(), pred_choice.cpu()))
    allr.append(recall_score(target.data.cpu(), pred_choice.cpu()))
    fpr, tpr, thresholds = roc_curve(target.data.cpu(),
                                     torch.sigmoid(pred.data).cpu(), pos_label=1)
    # allauc.append(auc(fpr, tpr))
    #print(roc_auc_score(target.data.cpu(), torch.sigmoid(pred.data).cpu()))
    allauc.append(roc_auc_score(target.data.cpu(), torch.sigmoid(pred.data).cpu()))
f = open(f"results_test_run{opt.fold}.txt", "a")
f.write(opt.fold + ',' + opt.model + ',' + str(sum(all) * 1.0 / len(all)) + ',' + str(sum(allp) * 1.0 / len(all)) + ',' + str(sum(allr) * 1.0 / len(all)) + ',' + str(sum(allauc) * 1.0 / len(all))+ "\n")
f.close()

print(sum(all) * 1.0 / len(all))
print(sum(allp) * 1.0 / len(all))
print(sum(allr) * 1.0 / len(all))
print(sum(allauc) * 1.0 / len(all))

