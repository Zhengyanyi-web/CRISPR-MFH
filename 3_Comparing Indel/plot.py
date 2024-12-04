import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os

def modellist():
    foldernames = [f for f in os.listdir('.') if os.path.isdir(f)]
    # 输出文件夹名数组
    a = []
    for foldername in foldernames:
        foldername = foldername.replace('_CIRCLE', '')
        if foldername not in a:
            a.append(foldername)
    return a

# Function to calculate mean precision and mean difference for a given model folder
def calculate_pr_curve_for_model(model_folder):
    all_precisions = []
    all_aucs = []

    # Define a common recall axis for interpolation
    common_recall_axis = np.linspace(0, 1, 100)

    # Loop through the 5 folds
    for fold in range(1, 6):
        # Construct the file path for each fold
        file_path = os.path.join(model_folder, f'fold_{fold}.csv')

        # Load the data
        df = pd.read_csv(file_path)

        # Extract scores and labels
        scores = df['Score']
        labels = df['Label']

        # Calculate precision, recall, and thresholds
        precision, recall, _ = precision_recall_curve(labels, scores)

        # Calculate AUC for this fold
        pr_auc = auc(recall, precision)
        all_aucs.append(pr_auc)

        # Interpolate precision values on the common recall axis
        interp_precision = np.interp(common_recall_axis, recall[::-1], precision[::-1])

        # Store precision for this fold
        all_precisions.append(interp_precision)

    # Convert to numpy array for easier manipulation
    all_precisions = np.array(all_precisions)

    # Calculate the mean precision across all folds
    mean_precision = np.mean(all_precisions, axis=0)

    # Calculate the mean difference (absolute difference between each fold and the mean)
    mean_differences = np.mean(np.abs(all_precisions - mean_precision), axis=0)

    return common_recall_axis, mean_precision, mean_differences, np.mean(all_aucs)

# Function to calculate mean ROC curve for a given model folder
def calculate_roc_curve_for_model(model_folder):
    all_fprs = []
    all_tprs = []
    all_aucs = []

    # Define a common FPR axis for interpolation
    common_fpr_axis = np.linspace(0, 1, 100)

    # Loop through the 5 folds
    for fold in range(1, 6):
        # Construct the file path for each fold
        file_path = os.path.join(model_folder, f'fold_{fold}.csv')

        # Load the data
        df = pd.read_csv(file_path)

        # Extract scores and labels
        scores = df['Score']
        labels = df['Label']

        # Calculate FPR, TPR, and thresholds
        fpr, tpr, _ = roc_curve(labels, scores)

        # Calculate AUC for this fold
        roc_auc = auc(fpr, tpr)
        all_aucs.append(roc_auc)

        # Interpolate TPR values on the common FPR axis
        interp_tpr = np.interp(common_fpr_axis, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure the curve starts at (0,0)

        # Store TPR for this fold
        all_tprs.append(interp_tpr)

    # Convert to numpy array for easier manipulation
    all_tprs = np.array(all_tprs)

    # Calculate the mean TPR across all folds
    mean_tpr = np.mean(all_tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1,1)

    return common_fpr_axis, mean_tpr, np.mean(all_aucs)

# List of model folders (modify this list based on your folder structure)
model_folders = ['CRISPR_MFH', 'CRISPR-BERT', 'CRISPR-MCA', 'CRISPR_IP', 'CRISPR_Net']
datasetName = 'CIRCLE'

# Initialize plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Loop through each model folder and plot its PR curve and ROC curve
for idx, model_folder in enumerate(model_folders):
    # PR Curve
    recall_axis, mean_precision, mean_differences, mean_auc = calculate_pr_curve_for_model(model_folder+'_'+datasetName)
    axes[0].plot(recall_axis, mean_precision, label=f'{model_folder} (AUC = {mean_auc:.2f})', lw=2)

    # ROC Curve
    fpr_axis, mean_tpr, mean_roc_auc = calculate_roc_curve_for_model(model_folder+'_'+datasetName)
    axes[1].plot(fpr_axis, mean_tpr, label=f'{model_folder} (AUC = {mean_roc_auc:.2f})', lw=2)

# Configure PR Curve plot
axes[0].set_xlabel('Recall', fontsize=14)
axes[0].set_ylabel('Precision', fontsize=14)
axes[0].set_title('(a) PR_AUC for different models on the D9 dataset', fontsize=16)
axes[0].legend(loc='lower left', fontsize=12)
axes[0].grid(False)

# Configure ROC Curve plot
axes[1].set_xlabel('False Positive Rate', fontsize=14)
axes[1].set_ylabel('True Positive Rate', fontsize=14)
axes[1].set_title('(b) ROC_AUC for different models on the D9 dataset', fontsize=16)
axes[1].legend(loc='lower right', fontsize=12)
axes[1].grid(False)

# Adjust layout and show plot
plt.tight_layout()

plt.savefig('./Indels.png', bbox_inches='tight', dpi=300, format='png')  # 保存热力图
plt.show()
