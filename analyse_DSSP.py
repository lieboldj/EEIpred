import sys
from Bio.PDB import PDBParser
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, levene, shapiro

def get_unique_residues(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_file)
    unique_residues = list()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()[1]
                unique_residues.append(residue_id)
    return unique_residues


if __name__ == "__main__":
    bool_print = False
    dataset = sys.argv[1] #"CONTACT, EPPIC, PISA"
    path_to_pdb = "PInet/data/exon/pdb/"
    path_to_map = "data_collection/uniprot_EnsemblExonPDB_map/"
    path_to_data = "data_collection/cv_splits/"
    rr_cutoff = 4
    pre_trained = ""
    if "pretrained" in dataset:
        pre_trained = "pretrained_"
        dataset = dataset.replace("pretrained_", "")
    elif "pre" in dataset:
        pre_trained = "pre_"
        dataset = dataset.replace("pre_", "")

    filename_pos = path_to_data + "{}/{}_positives.txt".format(dataset, dataset)
    filename_neg = path_to_data + "{}/{}_negatives.txt".format(dataset, dataset)

    df_pos = pd.read_csv(filename_pos, sep='\t')
    df_neg = pd.read_csv(filename_neg, sep='\t')

    mode = sys.argv[2] #"train, test"
    method = sys.argv[3] #"dMaSIF, PInet, glinter" ProteinMAE
    if len(sys.argv) > 5:
        method_path = sys.argv[5] #"dmasif, PInet, glinter" ../ProteinMAE/search
    elif method == "dMaSIF":
        method_path = "dmasif"
    elif method == "GLINTER":
        method_path = "glinter"
    else:
        method_path = method
    folds = sys.argv[4] #[1,2,3,4,5]" for all folds
    df_aa = pd.read_csv("data_collection/aa_interactions6_DSSP.txt", sep='\t')
    # rename last two columns to "AA1" and "AA2"
    df_aa.columns = ['chain1', 'chain2', 'exon1', 'exon2', 'AA1', 'AA2', 'DSSP1', 'DSSP2']
    #split folds at ,
    folds = folds.split(",")
    for i in folds:
        # use file with all protein pairs per dataset and folds combined
        df_test = pd.read_csv(f"data_collection/cv_splits/{dataset}/combined.txt", sep='_', header=None)
        df_testinfo = pd.read_csv(f"data_collection/cv_splits/{dataset}/combined_info.txt", sep='\t', header=None)
        df_test.columns = ['PDB', 'Chain1', 'Chain2']
        df_testinfo.columns = ['UniProt1', 'UniProt2']
        exon_pair_counter = 0
        exon_pair_pos_counter = 0
        exon_pair_neg_counter = 0
        inter_aa = []
        not_inter_aa = []
        poss = {}
        negs = {}
        pos_aa = {}
        if os.path.exists(f"{dataset}_all_exons.npy"):
            pass
        else:
            for j in tqdm(range(len(df_test))):
                pdb = df_test["PDB"][j]
                chain1 = df_test["Chain1"][j]
                chain2 = df_test["Chain2"][j]
                uniprot1 = df_testinfo["UniProt1"][j]
                uniprot2 = df_testinfo["UniProt2"][j]
                # check whether there is a map file for this PDB
                map_file1 = path_to_map + "{}_{}_{}.txt".format(uniprot1, pdb, chain1)
                map_file2 = path_to_map + "{}_{}_{}.txt".format(uniprot2, pdb, chain2)

                data1 = np.genfromtxt(map_file1, delimiter="\t", dtype=str, skip_header=1)

                # load exon2 information
                data2 = np.genfromtxt(map_file2, delimiter="\t", dtype=str, skip_header=1)

                # Initialize a regular dictionary to store the counts
                dssp1 = {}
                dssp2 = {}
                new_dssp1 = {}
                new_dssp2 = {}

                # Define the mapping for column[3] to the index in the list
                mapping = {"alpha": 0, "beta": 1, "N": 2, "non": 3}

                # only keep lines in data1 and data2 if line[-1] != "-"
                data1 = [line for line in data1 if line[-1] != "-"]
                data2 = [line for line in data2 if line[-1] != "-"]
                # Process each line of data
                #print(data1)
                for columns in data1:
                    dssp1.setdefault(str(columns[0]), [0, 0, 0, 0])[mapping.get(columns[3], 0)] += 1
                for columns in data2:
                    dssp2.setdefault(str(columns[0]), [0, 0, 0, 0])[mapping.get(columns[3], 0)] += 1

                # save each exon pair in protein_pair_random in npy file
                for exon1 in dssp1.keys():
                    for exon2 in dssp2.keys():
                        pos_bool = False
                        # check whether exon pair is in df_pos
                        if (df_pos[(df_pos["EXON1"] == exon1) & (df_pos["EXON2"] == exon2) & \
                        (df_pos["CHAIN1"] == (pdb+"_"+chain1)) & (df_pos["CHAIN2"] == pdb+"_"+chain2)].index.values).size > 0:
                            exon_pair_pos_counter += 1
                            pos_bool = True

                            # check whether the exon pair is in the aa_interactions file
                            if (df_aa[(df_aa["exon1"] == exon1) & (df_aa["exon2"] == exon2)].index.values).size > 0:
                                residue_pairs = df_aa[(df_aa["exon1"] == exon1) & (df_aa["exon2"] == exon2)]

                            elif (df_aa[(df_aa["exon1"] == exon2) & (df_aa["exon2"] == exon1)].index.values).size > 0:
                                residue_pairs = df_aa[(df_aa["exon1"] == exon2) & (df_aa["exon2"] == exon1)]

                        elif (df_pos[(df_pos["EXON1"] == exon2) & (df_pos["EXON2"] == exon1) & \
                        (df_pos["CHAIN1"] == pdb+"_"+chain2) & (df_pos["CHAIN2"] == pdb+"_"+chain1)].index.values).size > 0:
                            
                            # check whether the exon pair is in the aa_interactions file
                            if (df_aa[(df_aa["exon1"] == exon1) & (df_aa["exon2"] == exon2)].index.values).size > 0:
                                residue_pairs = df_aa[(df_aa["exon1"] == exon1) & (df_aa["exon2"] == exon2)]

                            elif (df_aa[(df_aa["exon1"] == exon2) & (df_aa["exon2"] == exon1)].index.values).size > 0:
                                residue_pairs = df_aa[(df_aa["exon1"] == exon2) & (df_aa["exon2"] == exon1)]

                            exon_pair_pos_counter += 1
                            pos_bool = True
                        else:
                            exon_pair_neg_counter += 1

                        # summarize these as a list of counts
                        #print(residue_pairs)
                        inter = [0, 0, 0, 0]
                        if pos_bool:
                            inter = [0, 0, 0, 0]
                            # Count occurrences for DSSP1
                            set1 = set(residue_pairs['AA1'])
                            set2 = set(residue_pairs['AA2'])
                            # iterate over complete dataframe residue_pairs
                            for index, row in residue_pairs.iterrows():
                                if pd.isna(row["DSSP1"]) or pd.isna(row["DSSP2"]):
                                    print(row)
                                    continue
                                if row["AA1"] in set1:
                                    inter[mapping[row["DSSP1"]]] += 1

                                    # remove the value from set1
                                    set1.remove(row["AA1"])
                                if row["AA2"] in set2:

                                    inter[mapping[row["DSSP2"]]] += 1
                                    
                                    # remove the value from set2
                                    set2.remove(row["AA2"])

                        if bool_print:
                            print("inter: ", inter)
                            print(sum(dssp1[exon1]), sum(dssp2[exon2]))

                        # add dssp1 and dssp2 element wise
                        dssp12 = [dssp1[exon1][0] + dssp2[exon2][0], dssp1[exon1][1] + dssp2[exon2][1], dssp1[exon1][2] + dssp2[exon2][2], dssp1[exon1][3] + dssp2[exon2][3]]

                        if pos_bool:
                            pos_aa[(exon1, exon2)] = inter
                            poss[(exon1, exon2)] = dssp12
                            
                        else:
                            negs[(exon1, exon2)] = dssp12
        # Prepare the data
        alpha_data = []
        beta_data = []
        n_data = []
        non_data = []
        labels = []
        colors = []

        if os.path.exists(f"{dataset}_a_b_n.csv"):
            df = pd.read_csv(f"{dataset}_a_b_n.csv")
            alpha_data = df["Alpha"].values
            beta_data = df["Beta"].values
            n_data = df["N"].values
            #non_data = df["Non"].values
            labels = df["Exon pairs"].values
            colors = df["Exon pairs"].values
        else:

            ################################
            # 3D scatter plot
            ################################
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            # merge N and non to one element for each list
            for key in poss.keys():
                poss[key] = poss[key][:2] + [poss[key][2] + poss[key][3]]
                # divide each by the sum of the list
                #poss[key] = poss[key] / np.sum(poss[key])
                ax.scatter(poss[key][0], poss[key][1], poss[key][2], c='red')
                alpha_data.append(poss[key][0])
                beta_data.append(poss[key][1])
                n_data.append(poss[key][2])
                #non_data.append(poss[key][3])
                labels.append('Interacting')
                colors.append('red')
            for key in negs.keys():
                negs[key] = negs[key][:2] + [negs[key][2] + negs[key][3]]
                #negs[key] = negs[key] / np.sum(negs[key])
                ax.scatter(negs[key][0], negs[key][1], negs[key][2], c='blue')

                alpha_data.append(negs[key][0])
                beta_data.append(negs[key][1])
                n_data.append(negs[key][2])
                #non_data.append(negs[key][3])
                labels.append('Non-interacting')
                colors.append('blue')

            ax.set_xlabel('alpha')
            ax.set_ylabel('beta')
            ax.set_zlabel('N+non')
            plt.legend(["interacting exon pairs", "non-interacting exon pairs", "interacting residue pairs from interacting exon pairs"])
            plt.savefig(f"{method}_{dataset}_{mode}_{i}.pdf")

            df = pd.DataFrame({
                'Alpha': alpha_data,
                'Beta': beta_data,
                'N': n_data,
                #'Non': non_data,
                'Exon pairs': labels
            })

        # save df
            df.to_csv(f"{dataset}_a_b_n.csv", index=False)

        print(len(df["Alpha"]))


        ################################
        # Violin plot
        ################################

        datasets = ["CONTACT", "PISA", "EPPIC"]
        fontsize = 16
        # Set the font size of the plots
        plt.rcParams.update({'font.size': fontsize})
        # Plotting the violin plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 9))
        
        for d, dataset in enumerate(datasets):
            df = pd.read_csv(f"{dataset}_a_b_n.csv")

            counter = 0
            df['Alpha'] = df['Alpha'].astype(float)
            df['Beta'] = df['Beta'].astype(float)
            df['N'] = df['N'].astype(float)
            # print the sum of each row
            epsilon = 1e-10
            for i in range(len(df)):
                alpha = df.loc[i,'Alpha']
                beta = df.loc[i,'Beta']
                N = df.loc[i,'N']

                sum_row = alpha + beta + N

                df.loc[i,'Alpha'] = alpha / sum_row
                df.loc[i,'Beta'] = beta / sum_row
                df.loc[i,'N'] = N / sum_row

                # if the sum is not 1, print the row because it is not normalized
                if abs(df['Alpha'][i] + df['Beta'][i] + df['N'][i] - 1) > epsilon:
                    counter += 1
                    exit()

            print("counter: ", counter)

            # print the mean of each column
            print("mean alpha: ", df["Alpha"].mean(), df["Alpha"].min(), df["Alpha"].max())
            print("mean beta: ", df["Beta"].mean(), df["Beta"].min(), df["Beta"].max())
            print("mean N: ", df["N"].mean(), df["N"].min(), df["N"].max())

            print("dataset: ", dataset)
            print("interacting exon pairs: ", len(df[df["Exon pairs"] == "Interacting"]))
            print("non-interacting exon pairs: ", len(df[df["Exon pairs"] == "Non-interacting"]))

            # Use 'hue' to specify categories and 'palette' to define colors
            sns.violinplot(x='Exon pairs', y='Alpha', data=df, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[0,d], cut=0, legend=False)
            sns.violinplot(x='Exon pairs', y='Beta', data=df, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[1,d], cut=0, legend=False)
            sns.violinplot(x='Exon pairs', y='N', data=df, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[2,d], cut=0, legend=False)
            
            # set ylim
            axes[0,d].set_ylim(0, 1.05)
            axes[1,d].set_ylim(0, 1.05)
            axes[2,d].set_ylim(0, 1.05)
            axes[0,d].set_ylabel('')
            axes[1,d].set_ylabel('')
            axes[2,d].set_ylabel('')
            # remove xlabels
            axes[0,d].set_xlabel('')
            axes[1,d].set_xlabel('')
            axes[2,d].set_xlabel('')

            # analyse statistically the differences between interacting and non-interacting exon pairs
            alpha_interacting = df[df["Exon pairs"] == "Interacting"]["Alpha"].values
            alpha_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["Alpha"].values
            beta_interacting = df[df["Exon pairs"] == "Interacting"]["Beta"].values
            beta_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["Beta"].values
            N_interacting = df[df["Exon pairs"] == "Interacting"]["N"].values
            N_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["N"].values

            # calculate t-test
            t_alpha, p_alpha = ttest_ind(alpha_interacting, alpha_non_interacting)
            t_beta, p_beta = ttest_ind(beta_interacting, beta_non_interacting)
            t_N, p_N = ttest_ind(N_interacting, N_non_interacting)
            print("t_alpha: ", t_alpha, "p_alpha: ", p_alpha)
            print("t_beta: ", t_beta, "p_beta: ", p_beta)
            print("t_N: ", t_N, "p_N: ", p_N)

            # calculate Mann-Whitney U test
            u_alpha, p_alpha = mannwhitneyu(alpha_interacting, alpha_non_interacting)
            u_beta, p_beta = mannwhitneyu(beta_interacting, beta_non_interacting)
            u_N, p_N = mannwhitneyu(N_interacting, N_non_interacting)
            print("u_alpha: ", u_alpha, "p_alpha: ", p_alpha)
            print("u_beta: ", u_beta, "p_beta: ", p_beta)
            print("u_N: ", u_N, "p_N: ", p_N)

            # add p-values to the plot
            axes[0,d].text(0.5, 1.2, f"Mann-Whitney U test (p-value): {p_alpha:.2e}", \
                           horizontalalignment='center', verticalalignment='center', transform=axes[0,d].transAxes, fontsize=fontsize-2)
            axes[1,d].text(0.5, 1.2, f"Mann-Whitney U test (p-value): {p_beta:.2e}", \
                           horizontalalignment='center', verticalalignment='center', transform=axes[1,d].transAxes, fontsize=fontsize-2)
            axes[2,d].text(0.5, 1.2, f"Mann-Whitney U test (p-value): {p_N:.2e}", \
                           horizontalalignment='center', verticalalignment='center', transform=axes[2,d].transAxes, fontsize=fontsize-2)

            # calculate Kolmogorov-Smirnov test
            ks_alpha, p_alpha = ks_2samp(alpha_interacting, alpha_non_interacting)
            ks_beta, p_beta = ks_2samp(beta_interacting, beta_non_interacting)
            ks_N, p_N = ks_2samp(N_interacting, N_non_interacting)

            print("ks_alpha: ", ks_alpha, "p_alpha: ", p_alpha)
            print("ks_beta: ", ks_beta, "p_beta: ", p_beta)
            print("ks_N: ", ks_N, "p_N: ", p_N)

            # calculate Levene test - check for equal variance
            levene_alpha, p_alpha_std = levene(alpha_interacting, alpha_non_interacting)
            levene_beta, p_beta_std = levene(beta_interacting, beta_non_interacting)
            levene_N, p_N_std = levene(N_interacting, N_non_interacting)

            print("levene_alpha: ", levene_alpha, "p_alpha: ", p_alpha)
            print("levene_beta: ", levene_beta, "p_beta: ", p_beta)
            print("levene_N: ", levene_N, "p_N: ", p_N)


            # add p-values to the plot but below the violin plot
            axes[0,d].text(0.5, 1.05, f"Kolmogorov-Smirnov test (p-value): {p_alpha:.2e}", \
                           horizontalalignment='center', verticalalignment='center', transform=axes[0,d].transAxes, fontsize=fontsize-2)
            axes[1,d].text(0.5, 1.05, f"Kolmogorov-Smirnov test (p-value): {p_beta:.2e}", \
                           horizontalalignment='center', verticalalignment='center', transform=axes[1,d].transAxes, fontsize=fontsize-2)
            axes[2,d].text(0.5, 1.05, f"Kolmogorov-Smirnov test (p-value): {p_N:.2e}", \
                           horizontalalignment='center', verticalalignment='center', transform=axes[2,d].transAxes, fontsize=fontsize-2)

            # increase distance between subplots
            plt.subplots_adjust(hspace=1)

            
        axes[0,0].set_ylabel('$\mathbf{α-helix}$\noccurences')
        axes[1,0].set_ylabel('$\mathbf{β-sheet}$\noccurences')
        axes[2,0].set_ylabel('$\mathbf{coil-turn}$\noccurences')

        # add xlabels to the last row
        axes[2,0].set_xlabel('Exon pairs')
        axes[2,1].set_xlabel('Exon pairs')
        axes[2,2].set_xlabel('Exon pairs')

        # add title above the text
        axes[0,0].text(0.5, 1.4, '$D_{Con}$', horizontalalignment='center', verticalalignment='center', transform=axes[0,0].transAxes, fontsize=fontsize +2)
        axes[0,1].text(0.5, 1.4, '$D_{Engy}$', horizontalalignment='center', verticalalignment='center', transform=axes[0,1].transAxes, fontsize=fontsize +2)
        axes[0,2].text(0.5, 1.4, '$D_{Evol}$', horizontalalignment='center', verticalalignment='center', transform=axes[0,2].transAxes, fontsize=fontsize +2)
        
        # Adjust layout and save the plot
        plt.tight_layout()
        #plt.savefig(f"plots/{dataset}_violin_norm.pdf")
        plt.savefig(f"plots/{dataset}_violin_norm.png", dpi=900)

        # 
        plt.close()
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 9))

        for d, dataset in enumerate(datasets):
            df = pd.read_csv(f"{dataset}_a_b_n.csv")
            # devide for each row the columns alpha, beta, N by the sum of the row
            df["Alpha"] = df["Alpha"] / (df["Alpha"] + df["Beta"] + df["N"])
            df["Beta"] = df["Beta"] / (df["Alpha"] + df["Beta"] + df["N"])
            df["N"] = df["N"] / (df["Alpha"] + df["Beta"] + df["N"])

            # plot histograms but only show the kde line and not the histogram
            sns.kdeplot(data=df, x='Alpha', hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[0,d], common_norm=True, fill=True)
            sns.kdeplot(data=df, x='Beta', hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[1,d], common_norm=True, fill=True)
            sns.kdeplot(data=df, x='N', hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[2,d], common_norm=True, fill=True)

        plt.tight_layout()
        plt.savefig(f"plots/{dataset}_histograms_norm.pdf")



        # print how many interacting and non-interacting exon pairs are in the dataset
        print("interacting exon pairs: ", len(df[df["Exon pairs"] == "Interacting"]))
        print("non-interacting exon pairs: ", len(df[df["Exon pairs"] == "Non-interacting"]))
        