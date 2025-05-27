import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, levene, shapiro
from statsmodels.stats.multitest import multipletests

datasets = ["CLUST_CONTACT", "CLUST_PISA", "CLUST_EPPIC"]
#datasets = ["CONTACT", "PISA", "EPPIC"]
if __name__ == "__main__":
    ################################
    # Violin plot
    ################################

    fontsize = 20
    # Set the font size of the plots
    plt.rcParams.update({'font.size': fontsize})
    # Plotting the violin plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))

    all_mw = []
    all_ks = []
    # get the q-values first by iterating over all datasets
    for d, dataset in enumerate(datasets):
        df = pd.read_csv(f"{dataset}_a_b_n_test.csv")

        counter = 0
        df['Alpha'] = df['Alpha'].astype(float)
        df['Beta'] = df['Beta'].astype(float)
        df['N'] = df['N'].astype(float)

        int_pairs = np.load(f"{dataset}_IDR_pos_names_a.npy.npz")
        non_int_pairs = np.load(f"{dataset}_IDR_neg_names_a.npy.npz")
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

        # analyse statistically the differences between interacting and non-interacting exon pairs
        alpha_interacting = df[df["Exon pairs"] == "Interacting"]["Alpha"].values
        alpha_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["Alpha"].values
        beta_interacting = df[df["Exon pairs"] == "Interacting"]["Beta"].values
        beta_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["Beta"].values
        N_interacting = df[df["Exon pairs"] == "Interacting"]["N"].values
        N_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["N"].values

        # calculate Mann-Whitney U test
        u_alpha, p_alpha = mannwhitneyu(alpha_interacting, alpha_non_interacting)
        u_beta, p_beta = mannwhitneyu(beta_interacting, beta_non_interacting)
        u_N, p_N = mannwhitneyu(N_interacting, N_non_interacting)
        u_IDR, p_IDR = mannwhitneyu(int_pairs['IDR'], non_int_pairs['IDR'])
        
        # apply the benjamini-hochberg procedure
        all_mw.append(p_alpha)
        all_mw.append(p_beta)
        all_mw.append(p_N)
        all_mw.append(p_IDR)

        # calculate Kolmogorov-Smirnov test
        ks_alpha, p_alpha = ks_2samp(alpha_interacting, alpha_non_interacting)
        ks_beta, p_beta = ks_2samp(beta_interacting, beta_non_interacting)
        ks_N, p_N = ks_2samp(N_interacting, N_non_interacting)
        ks_IDR, p_IDR = ks_2samp(int_pairs['IDR'], non_int_pairs['IDR'])
        all_ks.append(p_alpha)
        all_ks.append(p_beta)
        all_ks.append(p_N)
        all_ks.append(p_IDR)

    # apply benjamini-hochberg procedure
    reject, q_values_mw, _, _ = multipletests(all_mw, method='fdr_bh')
    reject, q_values_ks, _, _ = multipletests(all_ks, method='fdr_bh')

    num_exon_int = []
    num_exon_non_int = []
    
    # plot the violin plots
    for d, dataset in enumerate(datasets):
        #df = pd.read_csv(f"plots/paper/{dataset}_a_b_n.csv")
        df = pd.read_csv(f"{dataset}_a_b_n_test.csv")
        int_pairs = np.load(f"{dataset}_IDR_pos_names_a.npy.npz")
        non_int_pairs = np.load(f"{dataset}_IDR_neg_names_a.npy.npz")

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

        print("dataset: ", dataset)
        print("interacting exon pairs: ", len(df[df["Exon pairs"] == "Interacting"]))
        print("non-interacting exon pairs: ", len(df[df["Exon pairs"] == "Non-interacting"]))

        # combine int_pairs and non_int_pairs in df_IDR
        df_IDR = pd.DataFrame({
            'IDR': np.concatenate((int_pairs['IDR'], non_int_pairs['IDR'])),
            'Exon pairs' : np.concatenate((np.repeat("Interacting", len(int_pairs['IDR'])), np.repeat("Non-interacting", len(non_int_pairs['IDR']))))
        })

        # Use 'hue' to specify categories and 'palette' to define colors
        sns.violinplot(x='Exon pairs', y='Alpha', data=df, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[0,d], cut=0, legend=False)
        sns.violinplot(x='Exon pairs', y='Beta', data=df, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[1,d], cut=0, legend=False)
        sns.violinplot(x='Exon pairs', y='N', data=df, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[2,d], cut=0, legend=False)
        sns.violinplot(x='Exon pairs', y='IDR', data=df_IDR, hue='Exon pairs', palette={'Interacting': 'green', 'Non-interacting': 'yellow'}, ax=axes[3,d], cut=0, legend=False)

        # set ylim
        for i in range(4):
            axes[i,d].set_ylim(0, 1.05)
            axes[i,d].set_yticks([0, 0.5, 1])
            axes[i,d].set_yticklabels([0, 0.5, 1])
            axes[i,d].set_ylabel('')
            axes[i,d].set_xlabel('')

        # analyse statistically the differences between interacting and non-interacting exon pairs
        alpha_interacting = df[df["Exon pairs"] == "Interacting"]["Alpha"].values
        alpha_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["Alpha"].values
        beta_interacting = df[df["Exon pairs"] == "Interacting"]["Beta"].values
        beta_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["Beta"].values
        N_interacting = df[df["Exon pairs"] == "Interacting"]["N"].values
        N_non_interacting = df[df["Exon pairs"] == "Non-interacting"]["N"].values

        print(dataset)
        print(len(alpha_interacting), len(alpha_non_interacting))
        print(len(beta_interacting), len(beta_non_interacting))
        print(len(N_interacting), len(N_non_interacting))
        print(len(int_pairs['IDR']), len(non_int_pairs['IDR']))
        num_exon_int.append(len(alpha_interacting))
        num_exon_non_int.append(len(alpha_non_interacting))

        # add p-values to the plot
        for i in range(4):
            axes[i,d].text(0.5, 1.35, f"M-W U test (q-value): {q_values_mw[d*4+i]:.2e}", \
                           horizontalalignment='center', verticalalignment='center', \
                            transform=axes[i,d].transAxes, fontsize=fontsize-2)
            axes[i,d].text(0.5, 1.1, f"K-S test (q-value): {q_values_ks[d*4+i]:.2e}", \
                       horizontalalignment='center', verticalalignment='center',\
                          transform=axes[i,d].transAxes, fontsize=fontsize-2)

        # increase distance between subplots
        plt.subplots_adjust(hspace=1)

    axes[0,0].set_ylabel('Fraction of\nexon pair\nresidues in\n$\mathbf{α-helix}$')
    axes[1,0].set_ylabel('Fraction of\nexon pair\nresidues in\n$\mathbf{β-sheet}$')
    axes[2,0].set_ylabel('Fraction of\nexon pair\nresidues in\n$\mathbf{coil-turn}$')
    axes[3,0].set_ylabel('Fraction of\nexon pair\nresidues in\n$\mathbf{IDR}$')

    # add xlabels to the last row
    for i in range(3):
        #axes[3,i].set_xlabel('Exon pairs')
        axes[3,i].set_xlabel(f'Exon pairs\n\
            Total sample sizes:               \n\
            {num_exon_int[i]} interacting and               \n\
            {num_exon_non_int[i]} non-interacting               ', fontsize=fontsize)


    # add title above the text
    axes[0,0].text(0.5, 1.6, '$D_{Con}$', horizontalalignment='center', verticalalignment='center', transform=axes[0,0].transAxes, fontsize=fontsize +4)
    axes[0,1].text(0.5, 1.6, '$D_{Engy}$', horizontalalignment='center', verticalalignment='center', transform=axes[0,1].transAxes, fontsize=fontsize +4)
    axes[0,2].text(0.5, 1.6, '$D_{Evol}$', horizontalalignment='center', verticalalignment='center', transform=axes[0,2].transAxes, fontsize=fontsize +4)

    plt.tight_layout()
    #plt.savefig(f"plots/{dataset}_violin_norm_Aug.pdf")
    plt.savefig(f"Fig4.png", dpi=600)
 
    plt.close()
  