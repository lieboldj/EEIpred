import numpy as np
import pandas as pd
datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
modes = ["DL","AA", "AA_4", "AA_8"]
sets = ["train", "val", "test"]
output_file = "data_stat_supp1.csv"

# check how many pairs in CONTACT_PDBs.txt are in the train, val and test set
def filter_pairs(clusters, pairs):
    selected_pairs = []
    # make a set of the clusters
    clusters = set(clusters)
    for pair in pairs:
        # if the first element of the pair is in the set of clusters, yield the pair
        pair = pair.split("_")
        if pair[0] + "_" + pair[1] in clusters and pair[0] + "_" + pair[2] in clusters:
            selected_pairs.append(pair)
    return selected_pairs
data_path = "../data_collection/scripts/"
with open(output_file, "w") as f:
    for dataset in datasets:
        dataset_clust = dataset.split("_")[1]
        # Load MMseqs2 clustering output
        clusters_df = pd.read_csv(f"{data_path}{dataset_clust}03_cluster.tsv", sep="\t", header=None, names=["Cluster", "Chain"])
        clusters_df["Chain"] = clusters_df["Chain"].apply(lambda x: x.split(".")[0])
        clusters_df["Cluster"] = clusters_df["Cluster"].apply(lambda x: x.split(".")[0])

        # only add the unique values of the column "Cluster" to the list
        clusters = clusters_df["Cluster"].unique().tolist()
        print("Total Clusters:", len(clusters))

        # exchange the clusters name with a list of the chains e.g. cluster_1 -> [chain1-chain2-chain3]
        clusters = [clusters_df[clusters_df["Cluster"] == cluster]["Chain"].tolist() for cluster in clusters]

        # Stratified sampling with forward-backward assignment
        sorted_clusters = sorted(clusters, key=len)
        num_folds = 5
        # assign clusters to 5 folds based on size
        folds = [[] for _ in range(5)]
        number_clust = [[] for _ in range(5)]
        for i, cluster in enumerate(sorted_clusters):
            if (i // num_folds) % 2 == 0:  # Forward direction
                fold_idx = i % num_folds
            else:  # Backward direction
                fold_idx = num_folds - 1 - (i % num_folds)
            folds[fold_idx].extend(cluster)
            number_clust[fold_idx].append(cluster)

        # load all pairs and their info
        with open(f'{data_path}{dataset_clust}_combined.txt', 'r') as f_2, \
            open(f'{data_path}{dataset_clust}_combined_info.txt', 'r') as f_info:
                pairs = [line.strip() for line in f_2]
                infos = [line.strip() for line in f_info]
        # create dict with key: pair, value: info
        pair_info = dict(zip(pairs, infos))

        # Assign pairs to folds
        fold_pairs = [filter_pairs(fold, pairs) for fold in folds]

        train_pairs_all = [[] for _ in range(5)]
        # create train sets
        data_needed1 = {}
        data_needed2 = {}
        for mode in sets:
            data_needed1[mode] = []
            data_needed2[mode] = []
        for test_fold in range(num_folds):
            train_clusters = []
            for fold in range(num_folds):
                if fold != test_fold and fold != (test_fold + 1) % num_folds:
                    train_clusters.extend(folds[fold])
            train_pairs_all[test_fold] = filter_pairs(train_clusters, pairs)
            
            data_needed1["train"].append(f"{len(train_clusters)}")
            data_needed2["train"].append(f"{len(train_pairs_all[test_fold])}")
            data_needed1["val"].append(f"{len(folds[(test_fold + 1) % num_folds])}")
            data_needed2["val"].append(f"{len(fold_pairs[(test_fold + 1) % num_folds])}")
            data_needed1["test"].append(f"{len(folds[test_fold])}")
            data_needed2["test"].append(f"{len(fold_pairs[test_fold])}")


        for i in range(1,6):
            for j, modus in enumerate(sets): 
                for k, mode in enumerate(modes): 
                    
                    path_to_non = f"../results/dMaSIF_{mode}/{dataset}_{modus}_neg_fold{i}.npy"
                    path_to_int = f"../results/dMaSIF_{mode}/{dataset}_{modus}_pos_fold{i}.npy"

                    non_res_train = np.load(path_to_non, allow_pickle=True)
                    inter_res_train = np.load(path_to_int, allow_pickle=True)
                    if k == 0:
                        f.write(f"{data_needed1[modus][i-1]};{data_needed2[modus][i-1]};")
                    if mode == "DL":
                        f.write(f";{len(inter_res_train)};;{len(non_res_train)};")
                    else: 
                        f.write(f"{len(inter_res_train)};{len(non_res_train)};")
                f.write("\n")
            f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
    f.close()






