

#%%
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

#%%
parser = argparse.ArgumentParser(description='Test significance.')
parser.add_argument('-mh', '--method', type=str, default="dMaSIF,PInet,GLINTER", help='Methods to test')
parser.add_argument('-p', '--pp', type=str, default="AA,DL", help='Preprocessings to test')
parser.add_argument('-d', '--dataset', type=str, default="CONTACT,PISA,EPPIC", help='Datasets to test')
parser.add_argument('-s', '--sampling', type=int, default=0, help='Sampling on or off, is recommanded for AU-ROC')
parser.add_argument('-mr','--metric', type=str, default="F-score", help='Metric to plot, choose between AU-PRC, AU-ROC, F-score, Precision, Recall, for Precision and Recall you need to provide alpha')
args = parser.parse_args()
methods = args.method.split(",")
pps = args.pp.split(",")
datasets = args.dataset.split(",")
metric_string = args.metric
if args.sampling == 1:
    sampling_on = True
else:
    sampling_on = False

# check if metric_string is in the list of metrics
metrics = ["F-score", "Precision", "Recall"] #"AU-ROC", "AU-PRC", 
if metric_string not in metrics:
    print("Please choose a metric from the following list: ", metrics)
    exit()
all_methods = ["dMaSIF", "PInet", "GLINTER"]
all_pps = ["AA", "Max", "DL"]
all_datasets = ["CONTACT","PISA", "EPPIC"]
plt.figure(figsize=(6,6))

metric_marker = ["o", "p", "X", "*", "^", "X"]
method_color = ["#0000a7", "#008176", "#eecc16"]
dataset_density = [0.2,0.5,1]
   
for e, metric_string in enumerate(metrics):
    metric_check = True
    data = dict()        
    for alphas in [[0.01],[0.02],[0.03],[0.04],[0.05]]:
        # add header to file for wilcoxon. 
        with open(f"../results/pre-calc/{metric_string}_{alphas[0]}.csv", "w") as f:
            # add a string to csv file with values 0 to 14 devided by comma
            f.write("Method,")
            f.write(",".join([str(i) for i in range(15)]))
            f.write("\n")

        
        # load RRI results
        data = np.load(f"../results/pre-calc/{metric_string}_all_{alphas[0]}_AADLS.npy", allow_pickle=True).item()
        if metric_string == "Recall":
            print("threshold", alphas[0])

        data_Max = dict()
        data_DLs = dict()
        print(alphas)

        for key in data.keys():
            if "DLs" in key:
                data_DLs[key] = [item for sublist in data[key] for item in sublist]
                #print(data_DLs[key])
            else:
                data_Max[key] = [item for sublist in data[key] for item in sublist]

        if metric_check:            
            for m, method in enumerate(methods):
                data_list = list()
                #data_list.extend(string_add)
                for dataset in datasets:
                    label_plot_DL = f"{method} {dataset} - AA"
                    label_plot_DLs = f"{method} {dataset} - DLs"

                    data_list.extend(data_Max[label_plot_DL])
                with open(f"../results/pre-calc/{metric_string}_{alphas[0]}.csv", "a") as f:
                    f.write(method + "_AA,")
                    f.write(",".join([str(i) for i in data_list]))
                    f.write("\n")                   

        # load Max and DL results
        data = np.load(f"../results/pre-calc/{metric_string}_all_{alphas[0]}_MaxDLS.npy", allow_pickle=True).item()

        data_Max = dict()
        data_DLs = dict()

        for key in data.keys():
            if "DLs" in key:
                data_DLs[key] = [item for sublist in data[key] for item in sublist]
            else:
                data_Max[key] = [item for sublist in data[key] for item in sublist]

        if metric_check:            
            for m, method in enumerate(methods):
                data_list_max = list()
                #data_list_dl = list()
                #data_list.extend(string_add)
                for dataset in datasets:
                    label_plot_Max = f"{method} {dataset} - Max"
                    #label_plot_DLs = f"{method} {dataset} - DLs"

                    data_list_max.extend(data_Max[label_plot_Max])
                    #data_list_dl.extend(data_DLs[label_plot_DLs])

                with open(f"../results/pre-calc/{metric_string}_{alphas[0]}.csv", "a") as f:
                    f.write(method + "_Max,")
                    f.write(",".join([str(i) for i in data_list_max]))
                    f.write("\n") 

            for m, method in enumerate(methods):
                #data_list_max = list()
                data_list_dl = list()
                #data_list.extend(string_add)
                for dataset in datasets:
                    #label_plot_Max = f"{method} {dataset} - Max"
                    label_plot_DLs = f"{method} {dataset} - DLs"

                    #data_list_max.extend(data_Max[label_plot_Max])
                    data_list_dl.extend(data_DLs[label_plot_DLs])

                with open(f"../results/pre-calc/{metric_string}_{alphas[0]}.csv", "a") as f:
                    f.write(method + "_DL,")
                    f.write(",".join([str(i) for i in data_list_dl]))
                    f.write("\n")                  

                
        if metric_string == "AU-PRC" or metric_string == "AU-ROC":
            metric_check = False


