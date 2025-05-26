import pandas as pd
import sys

for dataset in ["CONTACT", "EPPIC", "PISA"]
    # Load MMseqs2 clustering output
    clusters_df = pd.read_csv(f"{dataset}03_cluster.tsv", sep="\t", header=None, names=["Cluster", "Chain"])
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
    len_check = [0] * num_folds
    for i, cluster in enumerate(sorted_clusters):
        if (i // num_folds) % 2 == 0:  # Forward direction
            fold_idx = i % num_folds
        else:  # Backward direction
            fold_idx = num_folds - 1 - (i % num_folds)
        folds[fold_idx].extend(cluster)
        len_check[fold_idx] += 1

    tot_clust = 0
    print("Total Clusters in each fold:")
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}: {len(fold)} PDB chains")
        tot_clust += len(fold)
    print("Total PDB chains in all folds:", tot_clust)

    combinations = [
        (2, 3, 4),
        (3, 4, 5),
        (1, 4, 5),
        (1, 2, 5),
        (1, 2, 3)
    ]
    # print the sum of the clusters in the combinations
    for i, comb in enumerate(combinations):
        sum_comb = 0
        for j in comb:
            sum_comb += len(folds[j-1])
        print(f"Sum of PDB chains in clusters in combination {i+1}: {sum_comb}")


    print("Total Clusters in each fold:")
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}: {len_check} clusters")

    # load all pairs and their info
    with open(f'../cv_splits/{dataset}/combined.txt', 'r') as f, \
        open(f'../cv_splits/{dataset}/combined_info.txt', 'r') as f_info:
            pairs = [line.strip() for line in f]
            infos = [line.strip() for line in f_info]
    # create dict with key: pair, value: info
    pair_info = dict(zip(pairs, infos))

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

    # Assign pairs to folds
    fold_pairs = [filter_pairs(fold, pairs) for fold in folds]
    sum_fold_pairs = 0
    print("Total pairs in each fold:")
    for i, fold in enumerate(fold_pairs):
        print(f"Fold {i+1}: {len(fold)} pairs")
        sum_fold_pairs += len(fold)
    print("Total pairs in all folds:", sum_fold_pairs)
    exit()
    # write the test pairs to a file
    for i, fold in enumerate(fold_pairs):
        with open(f'../cv_splits/CLUST_{dataset}/test{i+1}.txt', 'w') as f:
            with open(f'../cv_splits/CLUST_{dataset}/test_info{i+1}.txt', 'w') as f_info:
                with open(f'../cv_splits/CLUST_{dataset}/val{(i+1)%num_folds+1}.txt', 'w') as fval:
                    with open(f'../cv_splits/CLUST_{dataset}/val_info{(i+1)%num_folds+1}.txt', 'w') as f_info_val:
                        for pair in fold:
                            f.write(f"{pair[0]}_{pair[1]}_{pair[2]}\n")
                            f_info.write(f"{pair_info[f'{pair[0]}_{pair[1]}_{pair[2]}']}\n")
                            fval.write(f"{pair[0]}_{pair[1]}_{pair[2]}\n")
                            f_info_val.write(f"{pair_info[f'{pair[0]}_{pair[1]}_{pair[2]}']}\n")

    train_pairs_all = [[] for _ in range(5)]
    # create train sets
    for test_fold in range(num_folds):
        train_clusters = []
        for fold in range(num_folds):
            if fold != test_fold and fold != (test_fold + 1) % num_folds:
                train_clusters.extend(folds[fold])
        train_pairs_all[test_fold] = filter_pairs(train_clusters, pairs)

        print("Train Val Test and % of val in train:", len(train_pairs_all[test_fold]), len(fold_pairs[(test_fold + 1) % num_folds]),\
            len(fold_pairs[test_fold]), len(fold_pairs[(test_fold + 1) % num_folds])/len(train_pairs_all[test_fold]))

    write train_pairs_all to file
    for i, fold in enumerate(train_pairs_all):
        with open(f'../cv_splits/CLUST_{dataset}/train{i+1}.txt', 'w') as f,\
            open(f'../cv_splits/CLUST_{dataset}/train_info{i+1}.txt', 'w') as f_info:
                for pair in fold:
                    f.write(f"{pair[0]}_{pair[1]}_{pair[2]}\n")
                    f_info.write(f"{pair_info[f'{pair[0]}_{pair[1]}_{pair[2]}']}\n")

