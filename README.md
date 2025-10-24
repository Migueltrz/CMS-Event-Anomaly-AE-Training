# Autoencoder-Based Anomaly Detection Using CMS CERN Data

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-active-success)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20ACARUS-lightgrey)

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

## Core Workflow

### 1. Data Loading and Preprocessing

- The dataset is read from CERN’s secure data storage and preprocessed using `pandas` and `numpy`.
- Numerical variables are standardized to zero mean and unit variance using `StandardScaler`.
- Non-numerical columns are excluded automatically to maintain compatibility with PyTorch tensors.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

df = pd.read_csv('private_cms_data.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.select_dtypes('number'))
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
```

Data is then split into training and validation subsets:

```python
from torch.utils.data import DataLoader, TensorDataset, random_split

dataset = TensorDataset(X_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
```

---

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

## Environment Setup

To reproduce the environment:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

Optional for GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Recommended hardware:  
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
