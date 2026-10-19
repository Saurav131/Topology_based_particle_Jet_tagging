# Topology-Informed Jet Tagging using Persistent Homology

A scientific machine learning project exploring the use of **topological data analysis (TDA)** for quark–gluon jet classification. The pipeline transforms particle-jet point clouds into persistence images using persistent homology, then uses a convolutional neural network (CNN) to classify jets.

**Paper:** [Topology-Informed Jet Tagging using Persistent Homology](https://arxiv.org/abs/2601.01450)  
**Conference:** CHEP 2026, Bangkok

## Overview

Particle jets produced in high-energy proton–proton collisions contain information about the underlying particle interactions. This project investigates whether topological features can provide useful information for distinguishing quark-initiated jets from gluon-initiated jets.

Instead of directly classifying particle constituents, the approach represents each jet as a point cloud and applies persistent homology to extract multiscale geometric and topological information.

The resulting persistence diagrams are converted into fixed-size persistence images, which serve as inputs to a CNN classifier.

A central objective is to compare the discriminative information captured by two homology groups:

- **\(H_0\):** Tracks connected components and their merging across filtration scales.
- **\(H_1\):** Captures loop-like structures that emerge and disappear across filtration scales.

The study finds that the two representations have comparable individual classification performance, while combining them improves validation performance.

## Methodology

The project consists of three main stages.

### 1. Data preprocessing

The project uses the publicly available [HLS4ML jet dataset](https://zenodo.org/records/3602260).

Each jet is represented using constituent-level kinematic features, including relative transverse momentum, rotated pseudorapidity, and rotated azimuthal angle.

The preprocessing workflow reads the HDF5 dataset, extracts jet labels and constituent features, and organizes the data into CSV files for subsequent analysis.

### 2. Persistent homology and persistence images

For each jet, the constituent point cloud is processed using a Vietoris–Rips filtration. Persistent homology is computed using [`ripser`](https://github.com/scikit-tda/ripser), producing persistence diagrams for \(H_0\) and \(H_1\).

The diagrams are transformed into fixed-size persistence images using [`persim`](https://github.com/scikit-tda/persim). The resulting images encode topological features in a representation suitable for conventional machine learning architectures.

The implementation uses 40 × 40 pixel images for each homology group. The \(H_0\) and \(H_1\) representations can be combined to provide a two-channel input to the CNN.

### 3. CNN-based classification

A CNN implemented in PyTorch learns discriminative patterns from the persistence images.

The training workflow includes:

- Construction of the CNN architecture using convolutional layers, batch normalization, ReLU activations, and pooling.
- Training with cross-entropy loss and the Adam optimizer.
- Learning-rate scheduling and validation.
- Evaluation using classification metrics, including accuracy, precision, recall, F1 score, and ROC AUC.
- Optional experiment tracking using Weights & Biases in offline mode.

## Repository structure


.
├── Pre-processing.ipynb
├── Persistance_homology.ipynb
├── CNN-Binary.ipynb
├── Diagrams/
├── Diagrams.zip
└── readme.txt

| File | Purpose |
|---|---|
| `Pre-processing.ipynb` | Reads the HLS4ML HDF5 files, extracts constituent features and labels, and prepares CSV datasets. |
| `Persistance_homology.ipynb` | Computes persistence diagrams, generates persistence images, and saves the feature arrays for CNN training. |
| `CNN-Binary.ipynb` | Defines, trains, and evaluates the CNN classifier. |
| `Diagrams/` | Contains visual material associated with the project. |

## Requirements

The notebooks use Python and the following principal libraries:

- NumPy and Pandas
- h5py
- Ripser
- Persim
- Matplotlib
- PyTorch
- scikit-learn
- Weights & Biases
- Seaborn

Install the core dependencies in a Python environment:

```bash
pip install numpy pandas h5py ripser persim matplotlib
pip install torch scikit-learn wandb seaborn
```

A CUDA-compatible PyTorch installation can be used when a supported GPU is available. Otherwise, the CNN can run on a CPU, subject to available computational resources.

## Running the pipeline

Run the notebooks in the following order:

1. **Pre-processing:** Download the HLS4ML dataset and prepare the input data.
2. **Persistent homology:** Generate \(H_0\) and \(H_1\) persistence images and save the feature arrays.
3. **CNN training:** Load the saved arrays, train the classifier, and evaluate its performance.

The persistence-image notebook saves `X_persistence.npy` and `y_labels.npy`, which are subsequently loaded by the CNN notebook.

**Configuration note:** The notebooks currently contain machine-specific file paths that must be updated for your environment. Ensure that the data preprocessing and labels are configured consistently for binary quark–gluon classification before running the CNN notebook.

## Results

The accompanying paper reports results on **245,109 jets**, using an 80:20 training–validation split.

| Input representation | Validation AUC |
|---|---:|
| \(H_0\) persistence images | 0.76 |
| \(H_1\) persistence images | 0.75 |
| Combined \(H_0 + H_1\) images | Slightly higher than either individual representation |

These results suggest that both connected-component structure and loop-like topological features contain useful information for quark–gluon jet classification.

The reported results are from the accompanying study; they should not be interpreted as guaranteed outcomes of a fresh run with different preprocessing, data splits, or training settings.

## References

1. S. Mittal, *Topology-Informed Jet Tagging using Persistent Homology*, [arXiv:2601.01450](https://arxiv.org/abs/2601.01450).
2. J. Duarte et al., [HLS4ML jet dataset](https://zenodo.org/records/3602260).
3. U. Bauer, *Ripser: Efficient computation of Vietoris–Rips persistence barcodes*, [GitHub repository](https://github.com/scikit-tda/ripser).
4. [Persim: Persistence diagrams and images](https://github.com/scikit-tda/persim).

## Acknowledgments

This project used the publicly available HLS4ML jet dataset from CERN Open Data and computational resources at C-DAC Pune. Preliminary work was presented as a poster at CHEP 2026 in Bangkok.
