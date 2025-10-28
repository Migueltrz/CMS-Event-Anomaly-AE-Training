![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange?logo=pytorch)# Autoencoder-Based Anomaly Detection Using CMS CERN Data


![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-active-success)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20ACARUS-lightgrey)


---

## Table of Contents
<details>
  <summary>Click to expand</summary>

1. [Overview](#overview)
2. [Technical Description](#technical-description)
3. [Core Workflow and Architecture](#workflow-and-architecture)
4. [Results](#results)
5. [Environment Setup](#enviroment-setup)
6. [Execution](#execution)
7. [Dataset Notice](#dataset-notice)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

</details>

---


---

## Overview

This repository contains the implementation of an **Autoencoder-based model for anomaly detection** trained on **non-public CMS CERN data**.  
The model identifies **anomalous particle collision events** that may indicate phenomena beyond the Standard Model.  
The project was implemented entirely in **PyTorch**, optimized on the **ACARUS high-performance computing cluster**, and adapted to handle **large-scale high-energy physics data**.

> ⚠️ The dataset used in this project is **private** and obtained directly from the **CERN database**, therefore it **cannot be publicly distributed**.

---

## Technical Description

The primary code is in the Jupyter Notebook `train_AEs_pytorch.ipynb`.  
It includes **data preprocessing, neural network definition, model training, evaluation, and anomaly scoring**.  
The workflow is structured to facilitate reproducibility and reusability for other scientific datasets.

---

## Core Workflow and Architecture

### 1. Data Loading and Preprocessing

This stage handles **reading, structuring, and preparing** high-energy physics event data from CERN’s internal HDF5 files (`.h5`), which store structured arrays of variables per event.  
Unlike CSV data, these files hold **multi-dimensional arrays** corresponding to multiple detector variables.

---

#### **a. HDF5 Data Exploration**

The dataset is opened using `h5py`, allowing us to inspect its internal keys (datasets) and their dimensions.  
This helps confirm the available physical quantities and the structure of the input data before modeling.

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("ntuple_merged_3.h5", "r") as f:
    for key in f.keys():
        print(f"Dataset: {key}, Shape: {f[key].shape}, Type: {f[key].dtype}")

    print(f['nElectron'][:5])
    print(f['nMuon'][:5])
```
Each dataset corresponds to a physical observable such as:

* The number of reconstructed electrons (`nElectron`)
* The number of reconstructed muons (`nMuon`)
* Jet or invariant mass quantities (`A_Zmass`, `B_Zmass`, etc.)

#### **b. Event Distribution Analysis**
`Dataset_ID` is a categorical variable identifying the event origin for example, QCD background (1, 2) or Higgs-like signal (3, 4, ...). We verify how many events exist per dataset ID to ensure data balance.

```python
with h5py.File("ntuple_merged_3.h5", "r") as f:
    dataset_ids = f['Dataset_ID'][:]

unique_ids, counts = np.unique(dataset_ids.astype(int), return_counts=True)

plt.figure(figsize=(8, 5))
plt.bar(unique_ids, counts, width=0.6, color='steelblue', edgecolor='black')
plt.xlabel('Dataset_ID (integer)')
plt.ylabel('Count')
plt.title('Distribution of Dataset_IDs')
plt.xticks(unique_ids)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

#### **c. Feature and Label Extraction**
Specific physical variables are defined for use in the training phase:
```python
features = [
    "A_Zmass", "B_Zmass", "C_Zmass", "D_Zmass",
    "A_Dr_Z", "B_Dr_Z", "C_Dr_Z", "D_Dr_Z",
    "MET_pt"
]
labels = ['Dataset_ID']
```
The following function reads those arrays from the `.h5` file, returning two NumPy arrays:

* `feature_array` → event-by-variable matrix
* `label_array` → binary encoding (0 = background, 1 = signal)

```python
def get_features_labels(file_name):
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root, features[0]).shape[0]
    feature_array = np.zeros((nevents, len(features)))
    label_array = np.zeros((nevents, 2))

    for i, feat in enumerate(features):
        feature_array[:, i] = getattr(h5file.root, feat)[:]

    bkg_ids = {1, 2}
    dataset_id_array = getattr(h5file.root, labels[0])[:]
    label_array[:, 0] = np.isin(dataset_id_array, list(bkg_ids)).astype(int)
    label_array[:, 1] = 1 - label_array[:, 0]

    h5file.close()
    return feature_array, label_array
```
#### **d. Signal vs. Background Visualization**
A quick inspection of the distributions confirms which variables differentiate the two classes. We use Seaborn histograms to compare “signal” (events of interest) and “background” (QCD-like noise).

```python
import seaborn as sns
import pandas as pd

df = pd.DataFrame(feature_array, columns=features)
df['label'] = np.where(label_array[:, 1] == 1, 'Signal', 'Background')

for feat in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feat, hue='label', stat='density', element='step', common_norm=False)
    plt.title(f'Distribution of {feat} by Class')
    plt.xlabel(feat)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()
``` 
#### **e. Dataset Splitting and PyTorch Wrapping**
After verification, the feature matrix and labels are divided into training, validation, and test sets, maintaining randomization and class balance and define a minimal `Dataset` class for PyTorch:

````python


X_train, X_temp, _, y_temp = train_test_split(feature_array, label_array, test_size=0.2, random_state=42, shuffle=True)
X_valid, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
````

````python

class SensorDataset(Dataset):
    def __init__(self, dataset: np.array):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return torch.FloatTensor(self.dataset[index])
````


### 2. Model Architecture

The Autoencoder is a **symmetric neural network** composed of an encoder (compression) and decoder (reconstruction).  
Its purpose is to learn the most representative latent structure of standard (non-anomalous) events.

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
```

The latent space (`z`) typically has between **16–64 dimensions**, depending on the input complexity.

---

### 3. Training and Optimization

The model is trained using **Mean Squared Error (MSE)** between the original and reconstructed input.  
This encourages the network to minimize the reconstruction difference for normal events.

```python
model = Autoencoder(input_dim=X_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}")
```

Training progress can be visualized by tracking both training and validation loss curves.  
An early stopping mechanism halts training when overfitting is detected.

---

### 4. Anomaly Detection and Scoring

Once trained, the Autoencoder is used to reconstruct unseen data.  
The reconstruction error serves as an **anomaly score** — higher errors indicate abnormal events.

```python
model.eval()
reconstructions = model(X_tensor)
reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1)

threshold = reconstruction_errors.mean() + 3 * reconstruction_errors.std()
anomalies = (reconstruction_errors > threshold)
```

This simple statistical threshold identifies events likely to represent **rare physics signals or detector anomalies**.

---

### 5. Evaluation and Visualization

Performance evaluation uses:
- **Reconstruction Error Distribution** (histogram comparison between normal and anomalous events)
- **Scatter Plots** for variable correlation inspection
- **Latent Space Visualization** with PCA/t-SNE for dimensional reduction

```python
import matplotlib.pyplot as plt

plt.hist(reconstruction_errors.detach().numpy(), bins=50, alpha=0.7)
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()
```
---
## Results

After extensive training, the autoencoder successfully reconstructed most background-like events, while signal-like events showed higher reconstruction errors, suggesting distinct latent behavior.

Typical plots include:

* Loss Curve Evolution

The model successfully converged, as both the training and validation loss curves show a sharp initial drop and then stabilize (flatten out) after about $7.5$ epochs, indicating the autoencoder has finished learning1111.

* Latent Space Projection

The model successfully converged, as both the training and validation loss curves show a sharp initial drop and then stabilize (flatten out) after about $7.5$ epochs, indicating the autoencoder has finished learning1111.

* Reconstruction Error Distribution

This log-scale histogram confirms the anomaly detection strategy: "Ruido" (Background) has a very low reconstruction error, while "Señal" (Signal) exhibits a higher and more dispersed reconstruction error (the desired anomaly signature).

* Signal vs. Background Scatter Distribution

This is the most important and illustrative result. This scatter plot, which includes an anomaly Umbral $= 1300.00$ (Threshold) 1, shows that the vast majority of "Ruido" (Background) events are correctly clustered with a very low reconstruction error, well below the threshold2. The plot effectively separates the anomalies: out of 24 total anomalies, 15 were correctly identified as "Señal" (Signal) events above the threshold, while 9 "Ruido" (Background) events were falsely classified as anomalies (false positives)3. This highlights the model's success in anomaly detection, despite a small degree of misclassification.

These outcomes validate the autoencoder’s ability to identify statistically rare events that could correspond to physics beyond the Standard Model.


---

## Environment Setup

To ensure reproducibility and compatibility with CERN’s data handling and PyTorch-based workflows, it is recommended to create a **dedicated Python environment** using either `conda` or `venv`.  
The project depends on **machine learning, visualization, and HDF5 processing libraries**.

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

Optional for GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Hardware used:  
- **GPU:** NVIDIA with CUDA 11.8+  
- **Memory:** ≥16 GB RAM  
- **OS:** Linux (ACARUS cluster or Ubuntu 22.04 LTS)

---

## Execution

Run the notebook:

```bash
jupyter notebook train_AEs_pytorch.ipynb
```

The notebook sections are organized as follows:
1. Data Import and Preprocessing  
2. Model Definition and Hyperparameter Setup  
3. Training Loop and Validation Monitoring  
4. Anomaly Detection and Visualization  

Intermediate outputs (plots and loss curves) are automatically generated and saved locally.

---

## Dataset Notice

The dataset originates from **CMS experiment data at CERN**.  
It contains high-dimensional particle event variables that are **not publicly accessible** due to CERN’s data privacy policy.  
For replication purposes, the model can be trained using **CERN Open Data** or **synthetic datasets** with similar structure.

---

## License

This project is distributed under the **MIT License**.  
Refer to the `LICENSE` file for detailed terms.

---

## Acknowledgments

Development and testing were conducted on the **ACARUS high-performance computing cluster**.  
The project forms part of a research initiative in high-energy physics data analysis focusing on **unsupervised event classification and anomaly detection**.
