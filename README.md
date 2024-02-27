# The power and limits of protein 3D structure-based exon-exon interaction prediction
![Image](Figure2structure.png)

## Abstract
Alternative splicing (AS) effects on cellular functions can be captured by changes in the underlying protein-protein interactions (PPIs). Because AS results in the gain or loss of exons, existing methods for predicting AS-related PPI changes utilize known PPI interfacing exon-exon interactions (EEIs), which only cover ~5% of possible human PPIs. Hence, there is a need to extend the existing limited knowledge about EEIs to further advance the functional understanding of AS. In this study, we explore whether we can utilize existing computational PPI interface prediction (PPIIP) methods to predict EEIs instead of their original task of predicting residue-residue interactions (RRIs). We evaluate three recent state-of-the-art PPIIP methods on the RRI as well as the EEI prediction tasks using known protein complex structures, covering ~230,000 RRIs and ~27,000 EEIs. Our results provide the first evidence that existing PPIIP methods can be extended for the EEI prediction task, showing F-score, precision, and recall performances of up to ~38%, ~63%, and ~28%, respectively, with a false discovery rate of less than 5%. Our study provides insights into the power and limits of existing PPIIP methods to predict EEIs, which can guide future developments of computational methods for the EEI prediction task. We provide streamlined computational pipelines to the use of each of the three considered PPIIP methods for the EEI prediction task to be utilized by the scientific community. 

## Hardware requirements
Models have been trained on either a NVIDIA A40 or a single Tesla T4 GPU.

## Software prerequisites
Before you begin, ensure that you have the following prerequisites installed on your system:
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `pip` package manager
- `python` >=3.11

## Installation Steps
Given CUDA version 12.1, the following installation guide was tested.

### 1. Create conda environment
```bash
conda create -n eeip python=3.11
```

### 2. Activate conda environment
```bash
conda activate eeip
```

### Optional: Install C-compiler in environment
```bash
conda install -c conda-forge gxx
conda install -c conda-forge cxx-compiler
```

### 3. Install required packages from requirements.txt via pip
```bash
cat requirements.txt | xargs -L 1 pip install
```

### 4. Clone the repository
```bash
git clone https://gitlab.rrz.uni-hamburg.de/bax4015/eeip
cd eeip/
```

## Usage
Given two pdb files which are representing the pair of proteins for predicting the EEI:

### Adding hydrogen atoms to PDB files
If you have PDB files and need to add hydrogen atoms, you can use the software "reduce." Reduce is a program for adding hydrogens to a protein structure given the 3D coordinates of the heavy atoms. 

To install Reduce and learn how to use it, you can refer to the [Reduce GitHub repository](https://github.com/rlabduke/reduce) for installation instructions and usage guidelines.

```bash
reduce -build -Quiet XXX.pdb > XXX.pdb
```

### Prediction of EEIs
To predict the exon-exon interactions between 1efx_C and 1efx_A, run

```bash
./get_eeip.sh
```
You find the predictions in the folder which will be displayed.
For details regarding dmasif, we refer to the original [dMaSIF github](https://github.com/FreyrS/dMaSIF).

## License
ADD the LICENCE! Check LICENSE file!

## Reference
Liebold J, Del Moral Morales A, Manalastas-Cantos K, Tsoy O, Kurtz S, Baumbach J, Newaz K (2024). The power and limits of protein 3D structure-based exon-exon interaction prediction. [bioRxiv](https://www.biorxiv.org/).
