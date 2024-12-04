import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from math import pi

# 数据
labels = ['Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC']
models = {
    'CRISPR-MFH': [0.997, 0.501, 0.802, 0.364, 0.971, 0.537],
    'CRISPR-BERT': [0.997, 0.416, 0.742, 0.289, 0.975, 0.429],
    'CRISPR-DNT': [0.996, 0.441, 0.737, 0.315, 0.958, 0.482],
    'CRISPR-IP': [0.996, 0.415, 0.608, 0.315, 0.974, 0.467],
    'CRISPR-Net': [0.997, 0.458, 0.861, 0.312, 0.977, 0.516]
}

# 设置雷达图
num_vars = len(labels)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

# 颜色统一设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [2, 3]})

# 绘制雷达图
ax = axs[0]
ax = fig.add_subplot(1, 2, 1, polar=True)

for idx, (model, values) in enumerate(models.items()):
    values += values[:1]  # 闭合雷达图
    ax.plot(angles, values, linewidth=2, linestyle='-', label=model, color=colors[idx])
    ax.fill(angles, values, alpha=0.25, color=colors[idx])

# 设置标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
ax.set_rlabel_position(0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="#000", size=14)
ax.set_ylim(0, 1)
ax.set_title('(a) Radar plots of generalization performance for different models', size=18, color='black', y=1.03)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

# 计算所有模型的PR曲线
def calculate_pr_curve_for_all_models():
    all_precisions = []
    all_aucs = []
    model_names = ['CRISPR-MFH', 'CRISPR-BERT', 'CRISPR-DNT', 'CRISPR-IP', 'CRISPR-Net']

    # 定义一个通用的recall轴用于插值
    common_recall_axis = np.linspace(0, 1, 100)

    # 遍历所有CSV文件
    for name in model_names:
        # 加载数据
        df = pd.read_csv(f'{name}.csv')

        # 提取分数和标签
        scores = df['Score']
        labels = df['Label']

        # 计算precision, recall
        precision, recall, _ = precision_recall_curve(labels, scores)

        # 计算AUC
        pr_auc = auc(recall, precision)
        all_aucs.append(pr_auc)

        # 在通用recall轴上插值precision值
        interp_precision = np.interp(common_recall_axis, recall[::-1], precision[::-1])
        all_precisions.append(interp_precision)

    return common_recall_axis, np.array(all_precisions), all_aucs, model_names

# 初始化PR曲线图
ax = axs[1]

# 计算PR曲线
recall_axis, all_precisions, all_aucs, model_names = calculate_pr_curve_for_all_models()

# 绘制PR曲线
for idx, precision in enumerate(all_precisions):
    ax.plot(recall_axis, precision, label=f'{model_names[idx]} (AUC = {all_aucs[idx]:.2f})', lw=2, color=colors[idx])

# 设置PR曲线图标签和标题
ax.set_xlabel('Recall', fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_title('(b) PR_AUC values for generalization of different models', fontsize=18)
ax.legend(loc='lower left', fontsize=12)
ax.grid(False)

# 保存为高分辨率图片并显示
plt.tight_layout()
plt.savefig('Model_Generalization.png', dpi=300, bbox_inches='tight')
plt.show()