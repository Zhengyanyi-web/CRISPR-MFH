# CRISPR-MFH: Lightweight Hybrid Deep Learning Framework for CRISPR-Cas9 Off-Target Prediction

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-brightgreen)](https://github.com/Zhengyanyi-web/CRISPR-MFH)

**Title of paper**  
CRISPR-MFH: A Lightweight Hybrid Deep Learning Framework with Multi-Feature Encoding for Improved CRISPR-Cas9 Off-Target Prediction

---

## 📦 Repository Overview
This repository is the official implementation of the CRISPR-MFH model, providing:
- Multi-feature encoding module
- Lightweight hybrid neural network architecture
- Complete training/evaluation process
- One-click reproduction function of paper results

---

## 🛠️ Environment Configuration
```bash
pip install -r requirements.txt
```

## Model Highlights
| Model              | Parameters | Model size(KB) | ETA of one epoch | Hardware Required                 | Deployment Feasibility                     |
|--------------------|------------|----------------|------------------|------------------------------------|--------------------------------------------|
| CRISPR-BERT        | 10,175,277 | 119,501        | 13 minutes       | High-end GPU (A100/V100), TPU      | Unsuitable for edge/embedded deployment    |
| CRISPR-M           | 1,196,417  | 15,219         | 2 minutes        | Mid-to-high-end GPU (RTX 3060+)    | Suitable for GPU-based servers             |
| CRISPR-DNT         | 589,190    | 6,503          | 1.2 minutes      | Mid-end GPU (RTX 2060+)            | Suitable for GPU-based servers             |
| CRISPR-MFH (Ours)  | 125,030    | 1,677          | 20 seconds       | CPU / Embedded Devices             | Ideal for lightweight, real-time applications |


## Repository Structure
**0_Hyper-Parameter to 5_Vis:** Directories containing replication code for the results presented in the associated research paper. Running the plot.py script within each folder will generate the corresponding figures and analyses.​

**Encoding_List.py:** This module handles data encoding, transforming sequence pairs into vector representations suitable for input into deep learning models.​

**Model.py: **Defines the architecture of the CRISPR-MFH model.​

**Training_Model.py:** Script to train the CRISPR-MFH model. Ensure that your dataset is pre-encoded using Csv2pkl.py before initiating training.​

**Csv2pkl.py**: Utility to encode datasets from CSV format to PKL format, preparing them for model training.​

**requestment.txt**: Lists the necessary dependencies and libraries required to run the code in this repository.

## Installation and Setup
1. Clone the Repository:

```bash

git clone https://github.com/Zhengyanyi-web/CRISPR-MFH.git
```
2. Install Dependencies:

Navigate to the repository's root directory and install the required packages:

```bash
pip install -r requestment.txt
```
## Usage Instructions
3. Data Preparation:

Use the Csv2pkl.py script to convert your dataset from CSV to PKL format:

```bash
python Csv2pkl.py --input your_dataset.csv --output encoded_data.pkl
```
4. Model Training:

Train the CRISPR-MFH model using the pre-encoded dataset:

```bash
python Training_Model.py --data encoded_data.pkl
```
5. Results Reproduction:

To replicate the results from the research paper, navigate to the respective directories (0_Hyper-Parameter to 5_Vis) and execute the plot.py scripts:

```bash
cd 0_Hyper-Parameter
python plot.py
```
Repeat this process for each directory to generate all corresponding figures and analyses.

## Citation
f you find this repository useful for your research, please cite our paper:

```bash
Will be updated later

```
