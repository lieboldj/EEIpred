import numpy as np

pos = np.load("../results/dMaSIF_DL/CONTACT_pos_fold3.npy")
back = np.load("../results/dMaSIF_DL/CONTACT_back_fold3.npy")

alpha = 0.05

#plot histogram and add vertical line at cutoff
import matplotlib.pyplot as plt
bin_intervals = np.linspace(0, 1, 21)  # For example, bins from -3 to 3 with a step of 1

# plot hist with relative frequency using weights argument
hist1, bins1, _ = plt.hist(pos, color="b", weights=np.ones(len(pos)) / len(pos), bins=bin_intervals, label="Interacting exon pairs (test set)")
# add a 'best fit' line


hist2, bins2, _ = plt.hist(back, color="orange", weights=np.ones(len(back)) / len(back), bins=bin_intervals, alpha=0.8, label="Non-interacting exon pairs (training set)")
#n, bins, patches = plt.hist(pos, color="b", density=True, bins=10, label="interacting exon pairs (test set)")
#y = ((1 / (np.sqrt(2 * np.pi) * np.std(pos))) *
#     np.exp(-0.5 * (1 / np.std(pos) * (bins - np.mean(pos)))**2))
#plt.plot(bins, y, '--')

#plt.hist(back, color="orange", density=True, bins=10, alpha=0.8, label="non-interacting exon pairs (training set)")
#y = ((1 / (np.sqrt(2 * np.pi) * np.std(back))) *
#     np.exp(-0.5 * (1 / np.std(back) * (bins - np.mean(back)))**2))
#plt.plot(bins, y, '--')

cutoff = np.percentile(back, (1 - alpha) * 100)

bin_width1 = bins1[1] - bins1[0]
bin_width2 = bins2[1] - bins2[0]

# Calculate bin centers
bin_centers1 = bins1[:-1] + bin_width1 / 2
bin_centers2 = bins2[:-1] + bin_width2 / 2


# Set custom x-axis ticks to represent the intervals between bins
#plt.xticks(bin_centers1, ['{:.2f}-{:.2f}'.format(bins1[i], bins1[i+1]) for i in range(len(bins1)-1)], rotation=90)
plt.xticks(bin_centers2, ['{:.2f}-{:.2f}'.format(bins2[i], bins2[i+1]) for i in range(len(bins2)-1)], rotation=90)
plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1, label="Threshold, accepting 5% false positives")
#plt.grid(True)
plt.legend(loc="upper right")
plt.ylabel("Fraction of exon pairs", labelpad=15)
plt.xlabel("Predicted EEI scores", labelpad=15)
plt.savefig("../results/plots/example_hist_bin10.png", dpi=600, bbox_inches="tight")

