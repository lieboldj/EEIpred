#%%
import torch
import argparse
from torch import Tensor, dropout
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#%%
parser = argparse.ArgumentParser(description='Get exon predictions from AlexNet or Maximum')
parser.add_argument('-m', '--model', type=str, default="Maximum", help='Model to use: AlexNet or Maximum')
parser.add_argument('-i', '--index', type=int, default= 1, help='Index of the cross validation fold')
parser.add_argument('-p', '--path', type=str, default='../seg/alex_all-samples', help='Path to the model')
parser.add_argument('-d', '--data', type=str, default="", help='Path to the data')
parser.add_argument('-r', '--results', type=str, default="../../results", help='Path to the ground truth')
parser.add_argument('-e', '--exons', type=str, default="../../../data_collection/int_exon_pairs.txt", help='Path to the exon pairs')
args = parser.parse_args()

model_path = args.path + str(args.index) + ".pth"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# dataset class for 100x100 max size AA level data
class ExonDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.pairs = {}
        filename = args.exons
        with open(filename, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exon_name = row['exon1'] + "_" + row['exon2']
                if exon_name not in self.pairs:
                    self.pairs[exon_name] = 1

    def __getitem__(self, index):
        self.exons = np.load(self.data[index], allow_pickle=True)
        shape_1 = self.exons.shape[0]
        shape_2 = self.exons.shape[1]
        self.exonData = np.zeros((100, 100))
        self.exonData[:shape_1, :shape_2] = self.exons[:, :]
        exons = self.data[index].split('_')
        if exons[-2] + "_" + exons[-1][:-4] in self.pairs or \
            exons[-1][:-4] + "_" + exons[-2] in self.pairs:    
            labels = 1
        else:
            labels = 0
        return self.exonData, labels#, self.data[index][:-8]
    
    def __len__(self):
        return len(self.data)
    
# # model for 2D 100x100 images with a small and big kernel 
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
##
            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
##
            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            #nn.ReLU(True),
            #nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveMaxPool2d((6, 6))
        #self.maxpool = nn.MaxPool2d((5,5), (5,5))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            #nn.Linear(384 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = x.view(-1, 1, 100, 100)
        #x = self.maxpool(x)
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = torch.sigmoid(out)
        return out
    

#%%

cv_idx = args.index

# load all files from one folder with glob
#train_folder = f"../../results/fold_{cv_idx}/train/"
test_folder = f"{args.results}/{args.data}/fold_{cv_idx}/test/"
test_files = glob.glob(test_folder + "*.npy")
# create datasets
#trainDataset = ExonDataset(train_data)
testDataset = ExonDataset(test_files)
# create dataloaders
testLoader = DataLoader(testDataset, batch_size=1, shuffle=True)
#valLoader = DataLoader(valDataset, batch_size=128, shuffle=False)
if args.model == "AlexNet":
    model = AlexNet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    #save_path = f"models/2DMedian-{cv_idx}.pt"
    # train model
    pred = []
    true = []
    all_pred =  []
    pos = []
    neg = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(testLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            outputs = outputs.cpu().numpy()
            outputs = outputs.squeeze(1)
            if labels == 1:
                pos.append(outputs)
            else:
                neg.append(outputs)
            pred.extend(outputs)
            true.extend(labels.cpu().numpy().flatten())
        #np.save(f"back-fore/Alex/pos_Alex_PI_{args.index}", np.asarray(pos))
        #np.save(f"back-fore/Alex/back_Alex_PI_{args.index}", np.asarray(neg))
        #fpr, tpr, threshold = roc_curve(true,pred)
        #print(roc_auc_score(true,pred))
        #plt.plot(fpr,tpr,label=f"ROC-AUC score, fold {cv_idx}: {round(roc_auc_score(true,pred), 2)}")
        #print(np.sum(true))

elif args.model == "Maximum":
    pos = []
    neg = []
    for inputs, labels in tqdm(testLoader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = torch.max(inputs.flatten())
        outputs = outputs.cpu().numpy()
        if labels == 1:
            pos.append(outputs)
        else:
            neg.append(outputs)
    # save the predictions
    #np.save(f"back-fore/Max/pos_Max_PI_{cv_idx}", np.asarray(pos))
    #np.save(f"back-fore/Max/back_Max_PI_{cv_idx}", np.asarray(neg))