#%%
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import csv
import time
#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# dataset class for 100x100 max size AA level data
class ExonDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.pairs = {}
        filename = "../data/int_exon_pairs.txt"
        with open(filename, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exon_name = row['exon1'] + "_" + row['exon2']
                if exon_name not in self.pairs:
                    self.pairs[exon_name] = 1

    def __getitem__(self, index):
        self.exons = np.load(self.data[index], allow_pickle=True)
        #if (np.sum(self.exons)) == 0:
        #    print(self.data[index])
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
class ExonDatasetPISAEPPIC(Dataset):
    def __init__(self, data, filename = "../data/int_exon_pairs.txt"):
        self.data = data

        self.pairs = {}
        with open(filename, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exon_name = row['exon1'] + "_" + row['exon2']
                if exon_name not in self.pairs:
                    self.pairs[exon_name] = 1

    def __getitem__(self, index):
        self.exons = np.load(self.data[index])
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
data_path = "PISA_EEIN_0.5/"
model_folder = f"{data_path}Alex"
filename = "../data/PISA_EEIN_0.5_positives.txt"
'''
for i in range(5):
    cv_idx = i+1
    
    # load all files from one folder with glob
    train_folder = f"../results/{data_path}part_{cv_idx}/train/"
    #test_folder = f"../../results/fold_{cv_idx}/test/"
    search_pattern = train_folder + "*.npy"
    exclude_pattern = train_folder + ".*big.npy"
    # Use glob to get the list of file names
    train_data = [filename for filename in glob.glob(search_pattern) if not filename.endswith('big.npy')]

    #test_files = glob.glob(test_folder + "*.npy")
    
   
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    #negExons_train, negExons_val = train_test_split(negExons, test_size=0.2, random_state=42)
    # create datasets
    trainDataset = ExonDatasetPISAEPPIC(train_data, filename = filename)
    valDataset = ExonDatasetPISAEPPIC(val_data, filename = filename)

    # create dataloaders
    trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=128, shuffle=False)

    model = AlexNet()
    model = model.to(device)
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    save_path = f'../models/Alex/{data_path}all-samples{cv_idx}.pth'
    #save_path = f"models/2DMedian-{cv_idx}.pt"

    # train model
    epochs = 15
    trainLoss = []
    testLoss = []
    val_loss = []
    pos = 0
    neg = 0
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for inputs, labels in tqdm(trainLoader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs.float())#[:,0]

            loss = criterion(outputs.squeeze(1), labels.float())#.squeeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        #print(pos, neg)
        trainLoss.append(np.mean(np.asarray(train_loss)))
        model.eval()
        with torch.no_grad():
            val_loss = []
            for inputs, labels in tqdm(valLoader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs.float())#[:,0]
                
                #labels = labels.float().view(-1, 1)
                #labels = F.one_hot(labels.to(torch.int64), num_classes=2)
                loss = criterion(outputs.squeeze(1), labels.float())
                #loss = criterion(outputs, labels)#.squeeze(1).float())
                val_loss.append(loss.item())
            testLoss.append(np.mean(np.asarray(val_loss)))
        scheduler.step()
        if testLoss[-1] == min(testLoss):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, save_path)
        print("Epoch: %d, Train Loss: %.3f, Val Loss: %.3f" % (epoch, trainLoss[epoch], testLoss[epoch]))

# %%
'''
for mode in ["train", "test"]:
    for i in range(5):
        cv_idx = i+1
        # load all files from one folder with glob
        train_folder = f"../results/{data_path}part_{cv_idx}/{mode}/"
        #test_folder = f"../../results/fold_{cv_idx}/test/"
        search_pattern = train_folder + "*.npy"
        exclude_pattern = train_folder + ".*big.npy"
        # Use glob to get the list of file names
        test_data = [filename for filename in glob.glob(search_pattern) if not filename.endswith('big.npy')]

        testDataset = ExonDatasetPISAEPPIC(test_data, filename = filename)

        # create dataloaders
        testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

        model = AlexNet()
        model = model.to(device)
        # load model

        save_path = f'../models/Alex/{data_path}all-samples{cv_idx}.pth'
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        pos = list()
        neg = list()
        with torch.no_grad():
            model.eval()
            counter = 0
            for inputs, labels in tqdm(testLoader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs.float())
                # convert outputs to numpy at cpui
                outputs = outputs.cpu().numpy()[0]
                if labels == 1:
                    pos.append(outputs[0])
                else:
                    neg.append(outputs[0])

        pos = np.asarray(pos)
        neg = np.asarray(neg)
        # save to for evalaluation
        if mode == "test":
            np.save(f"../results/{data_path}part_{cv_idx}/{mode}/pos.npy", pos)
            np.save(f"../results/{data_path}part_{cv_idx}/{mode}/neg.npy", neg)

        
# %%
