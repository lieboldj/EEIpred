

#%%
import os
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

#%%
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-mh', '--method', type=str, default="dMaSIF,PInet,GLINTER,ProteinMAE", help='Methods to test')#dMaSIF,PInet,GLINTER,
parser.add_argument('-p', '--pp', type=str, default="AA", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CLUST_CONTACT,CLUST_PISA,CLUST_EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AUROC')
parser.add_argument('-mr','--metric', type=str, default="AUROC", help='Metric to plot, choose between AUPRC, AUROC, Fscore, Precision, Recall, for Precision and Recall you need to provide alpha')
parser.add_argument('-a','--alpha', type=float, default=0.05, help='Threshold for precision and recall')
parser.add_argument('-c','--conf_matrix', type=int, default=0, help='Confusion matrix')
parser.add_argument('-csv','--csv', type=int, default=1, help='Create csv file with all results')

args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
metric_string = args.metric
alphas = [args.alpha]
conf_matrix = args.conf_matrix
create_csv = args.csv
if args.sampling == 1:
    sampling_on = True
else:
    sampling_on = False

# check if metric_string is in the list of metrics
metrics = ["AUPRC", "AUROC", "Fscore", "Precision", "Recall", "MCC"]
if metric_string not in metrics:
    print("Please choose a metric from the following list: ", metrics)
    exit()

dict_groups = methods

#plt.figure(figsize=(6,6))
data = dict()   
use_all = True 
if os.path.exists(f"../results/plots/all_clust{alphas[0]}_4.npy") and use_all:
    data = np.load(f"../results/plots/all_clust{alphas[0]}_4.npy", allow_pickle=True).item()  
#print(data)

# Extract settings
# only keep the keys that have one "p" in it
pp = pps[0]
data = {key: value for key, value in data.items() if pp in key}

if create_csv:
    # map format to fit google sheet for one method
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05]
    with open(f"tables/{pp}_1.csv", "w") as f:
        for method in methods:
            f.write(f"{method}\n")
            for alpha in alphas: 
                if "CLUST" in datasets[0]:
                    if os.path.exists(f"../results/plots/all_clust{alpha}_4.npy") and use_all:
                        data = np.load(f"../results/plots/all_clust{alpha}_4.npy", allow_pickle=True).item()  
                elif "CONTACT" in datasets[0]:
                    if os.path.exists(f"../results/plots/all_{alpha}.npy") and use_all:
                        data = np.load(f"../results/plots/all_{alpha}.npy", allow_pickle=True).item()
                else:
                    print("problems exiting")
                    exit()
                f.write(f"Threshold: {alpha},\n")
                for dataset in datasets:
                    if "CONTACT" in dataset:
                        f.write(f"D_Con,")
                    elif "PISA" in dataset:
                        f.write(f"D_Engy,")
                    elif "EPPIC" in dataset:
                        f.write(f"D_Evol,")
                    for metric in ["MCC", "Fscore", "Precision", "Recall"]:
                        #print(data.keys())
                        key = f"{method} {dataset} - {pp} - {metric}"
                        if key in data:
                            values = data[key][:,0]
                            f.write(f"{values[0]},{values[1]},{values[2]},{values[3]},{values[4]},{np.nanmean(values)},,")
                    f.write("\n")
                f.write("\n")
    f.close()


    with open(f"tables/{pp}_2.csv", "w") as f:
        #f.write(",AU-ROC,,,,,,,AU-PRC,,\n")
        #f.write(",Fold1,Fold2,Fold3,Fold4,Fold5,Mean,,Fold1,Fold2,Fold3,Fold4,Fold5,Mean,\n\n")
        for method in methods:
            f.write(f"{method},\n") 
            if "CLUST" in datasets[0]:
                if os.path.exists(f"../results/plots/all_clust{alpha}_4.npy") and use_all:
                    data = np.load(f"../results/plots/all_clust{alpha}_4.npy", allow_pickle=True).item()  
            elif "CONTACT" in datasets[0]:
                if os.path.exists(f"../results/plots/all_{alpha}.npy") and use_all:
                    data = np.load(f"../results/plots/all_{alpha}.npy", allow_pickle=True).item()
            else:
                print("problems exiting")
                exit()
            for dataset in datasets:
                if "CONTACT" in dataset:
                    f.write(f"D_Con,")
                elif "PISA" in dataset:
                    f.write(f"D_Engy,")
                elif "EPPIC" in dataset:
                    f.write(f"D_Evol,")
                for metric in ["AUROC", "AUPRC"]:
                    #print(data.keys())
                    key = f"{method} {dataset} - {pp} - {metric}"
                    if key in data:
                        values = data[key]
                        f.write(f"{values[0]},{values[1]},{values[2]},{values[3]},{values[4]},{np.nanmean(values)},,")
                f.write("\n")
            f.write("\n")
    f.close()     
