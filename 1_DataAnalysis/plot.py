import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
# Sample data (CSV file contents)
import os
import numpy as np
from matplotlib.gridspec import GridSpec
# Define the directory path (assuming abc is in the current working directory)
directory_path = '../../Data/DataSets/GC'  # Change this path to your correct folder path

combined_df = []

# Loop through all files in the specified directory
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 过滤出label为1的行
        filtered_df = df[df['label'] == 1]
        # 将过滤后的数据添加到列表中
        combined_df.append(filtered_df)


# Optionally, you can concatenate all dataframes into one
final_df = pd.concat(combined_df, ignore_index=True)


# Adding a column for mismatch positions (where the sequences differ)
unique_on_seq_df = pd.DataFrame({'on_seq': final_df['on_seq'].unique()})

# Calculating GC content for the unique on_seq values
unique_on_seq_df['gc_content'] = unique_on_seq_df['on_seq'].apply(lambda seq: (seq.count('G') + seq.count('C')) / len(seq))
unique_on_seq_df['gc_content_percentage'] = unique_on_seq_df['gc_content'] * 100

# Displaying the result to the user



# 指定文件路径
file_path = '../../Data/DataSets/Mismatch/Hek293t.csv'

# 读取CSV文件
print(f'Starting analysis of {file_path}...')
datalist = pd.read_csv(file_path)

# 筛选label为1的行
datalist = datalist[datalist['label'] == 1]

# 转换为NumPy数组以便进一步操作
datalist = np.array(datalist)
data = datalist[:, 0:2]  # 提取序列对
label = datalist[:, 2]  # 提取标签

# 定义编码映射
encoding_map = {
    'AC': 0, 'AG': 1, 'AT': 2,
    'CA': 3, 'CG': 4, 'CT': 5,
    'GA': 6, 'GC': 7, 'GT': 8,
    'TA': 9, 'TC': 10, 'TG': 11,
}

# 初始化错配矩阵
mismatch_matrix = pd.DataFrame(0, index=range(12), columns=range(23))
keys_list = ["rA - dC", "rA - dG", "rA - dT", "rC - dA", "rC - dG", "rC - dT", "rG - dA", "rG - dC", "rG - dT", "rT - dA", "rT - dC", "rT - dG"]

# 计算错配数量矩阵
for i in range(len(data)):
    pair = data[i]
    seq1 = pair[0]  # RNA序列
    seq2 = pair[1]  # DNA序列
    for j in range(len(seq2)):  # 遍历每个位置
        a = seq1[j].upper()  # RNA碱基
        b = seq2[j].upper()  # DNA碱基
        if a == 'N':  # 跳过 'N'
            a = b
        if b == 'N':
            b = a

        if a != b:
            pair_key = a + b
            if pair_key in encoding_map:  # 只有在错配映射表中的碱基对才记录
                mismatch_matrix.loc[encoding_map[pair_key], j] += 1

# 调整矩阵行列的索引
mismatch_matrix.index = mismatch_matrix.index + 1
mismatch_matrix.columns = mismatch_matrix.columns + 1


# 创建一个16x9的图形对象
# 创建一个16x6的图形对象
plt.figure(figsize=(16, 6))

# 使用GridSpec管理布局，2行2列
# width_ratios=[3, 7] 表示第一列占30%，第二列占70%
gs = GridSpec(2, 2, width_ratios=[3, 7])

# 设置字体大小
label_font_size = 14
title_font_size = 16
tick_font_size = 12

# 1. 在(1,1)位置绘制直方图，占30%的宽度
ax1 = plt.subplot(gs[0, 0])
sns.histplot(unique_on_seq_df['gc_content_percentage'], bins=20, kde=True, ax=ax1)
ax1.set_title('(a) Distribution of GC Content', fontsize=title_font_size)
ax1.set_xlabel('GC Content (%)', fontsize=label_font_size)
ax1.set_ylabel('Frequency', fontsize=label_font_size)
# ax1.grid(False)

# 设置刻度标签的字体大小
ax1.tick_params(axis='x', labelsize=tick_font_size)
ax1.tick_params(axis='y', labelsize=tick_font_size)

# 2. 在(2,1)位置绘制箱线图，占30%的宽度
ax2 = plt.subplot(gs[1, 0])
sns.boxplot(x=unique_on_seq_df['gc_content_percentage'], ax=ax2)
ax2.set_title('(b) Boxplot of GC Content', fontsize=title_font_size)
ax2.set_xlabel('GC Content (%)', fontsize=label_font_size)

# 设置刻度标签的字体大小
ax2.tick_params(axis='x', labelsize=tick_font_size)
ax2.tick_params(axis='y', labelsize=tick_font_size)

# 3. 在(1,2)和(2,2)合并的位置绘制热力图，占70%的宽度
ax3 = plt.subplot(gs[:, 1])  # 使用 gs[:, 1] 表示合并两行的子图
sns.heatmap(mismatch_matrix, cmap='Blues', square=True, annot=False, fmt='d', cbar=True, cbar_kws={"fraction": 0.025}, ax=ax3)
ax3.set_title('(c) Mismatch Heatmap', fontsize=title_font_size)
ax3.set_xlabel('Mismatch Position', fontsize=label_font_size)
ax3.set_ylabel('Mismatch Type', fontsize=label_font_size)

# 设置刻度标签的字体大小
ax3.tick_params(axis='x', labelsize=tick_font_size)
ax3.tick_params(axis='y', labelsize=tick_font_size)

# 设置y轴的标签字体大小
ax3.set_yticklabels(keys_list, rotation=0, fontsize=tick_font_size)

# 调整子图之间的布局
plt.tight_layout()

# 保存图像
# plt.savefig(fname="DataAnalysis.svg", format="svg", bbox_inches="tight", dpi=300)
# plt.savefig(fname="DataAnalysis.tif", format="tif", bbox_inches="tight", dpi=300)
plt.savefig(fname="DataAnalysis.png", format="png", bbox_inches="tight", dpi=300)
# plt.savefig(fname="DataAnalysis.eps", format="eps", bbox_inches="tight", dpi=300)
# plt.savefig(fname="DataAnalysis.pdf", format="pdf", bbox_inches="tight", dpi=300)
# 显示图表
plt.show()