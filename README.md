# Autoencoder-Based Anomaly Detection Using CMS CERN Data

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange?logo=pytorch)
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

---
### 2. Model Architecture

The model is a **fully connected Autoencoder** implemented using **PyTorch Lightning**, designed for the reconstruction of particle-physics observables extracted from CMS data.  
It compresses multidimensional event-level features into a smaller latent representation and reconstructs them to detect anomalies based on reconstruction error.

```python
class Autoencoder(L.LightningModule):
    def __init__(self, in_dim):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, 16),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.Linear(64, in_dim)
        )

        self.training_losses = []
        self.validation_losses = []

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
````

---

### 3. Training and optimization
The model is trained using `Smooth L1 Loss` (Huber Loss) to improve robustness against outliers and reduce the influence of extreme reconstruction errors often caused by rare signal events.

````python
def training_step(self, batch, batch_idx):
    x = batch
    x_hat = self(x)
    loss = F.smooth_l1_loss(x_hat, x)
    self.log('train_loss', loss, prog_bar=True)
    return loss
````
The `AdamW optimizer` is used with a small weight decay to prevent overfitting:

````python
def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

````
Early stopping and checkpointing ensure the model stops training once convergence is reached.

````python
es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
cp = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
trainer = L.Trainer(callbacks=[es, cp], max_epochs=1000, accelerator="auto")
trainer.fit(model, train_dataloaders=ae_tdl, val_dataloaders=ae_vdl)

````
After training, the loss curves are plotted.

````python
plt.plot(model.training_losses, label="Training Loss")
plt.plot(model.validation_losses[1:], label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.savefig("figs/losscurv.pdf")
plt.show()
````

---

### 4. Anomaly Detection and Scoring

The Autoencoder reconstructs unseen data. The reconstruction error is computed for each event to determine its anomaly likelihood.

````python
pred = trainer.predict(model, tdl)
reco = np.mean((np.vstack(pred) - X_test) ** 2, axis=1)

````
* Low error → event resembles background (QCD-like).
* High error → event deviates from standard patterns (potential signal or anomaly).

A comparison between background and signal reconstruction errors is shown in logarithmic scale

````python
plt.hist(reco[ruido_idx], bins=50, alpha=0.3, label='Background')
plt.hist(reco[senal_idx], bins=50, alpha=0.6, label='Signal')
plt.yscale('log')
plt.xlabel('Reconstruction Error')
plt.ylabel('Events')
plt.legend()
plt.savefig("figs/REH.pdf")
plt.show()
````
---
### 5. Evaluation and Visualization

The Autoencoder’s performance is quantified through:

 ROC curve to measure discriminative power between signal and noise.
  
````python
fpr, tpr, thresholds = roc_curve(y_test[:, 1], reco)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("figs/rocurv.pdf")
plt.show()

````

PCA and t-SNE projections to explore the latent space separation.

````python
# PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(model.encoder(torch.FloatTensor(X_test)).detach().numpy())
plt.scatter(proj[:, 0], proj[:, 1], c=y_test.argmax(axis=1), cmap='coolwarm', alpha=0.6)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA of Latent Space')
plt.savefig("figs/PCA.pdf")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
embed = tsne.fit_transform(proj)
plt.scatter(embed[:, 0], embed[:, 1], c=y_test.argmax(axis=1), cmap='viridis', alpha=0.6)
plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2'); plt.title('t-SNE of Latent Space')
plt.savefig("figs/SNE.pdf")
plt.show()
````

Scatter plots to visualize reconstruction error per event.

````python
threshold = 1300
plt.figure(figsize=(27,8))
plt.scatter(np.arange(len(reco)), reco, c=y_test[:,1], cmap='coolwarm', s=100, alpha=1)
plt.axhline(threshold, color='black', linestyle='--', linewidth=2)
plt.xlabel('Event Number')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error Distribution')
plt.legend(['Threshold', 'Background', 'Signal'])
plt.savefig("figs/scat.pdf")
plt.show()

````
  

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
