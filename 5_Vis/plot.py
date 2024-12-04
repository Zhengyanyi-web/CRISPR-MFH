import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import seaborn as sns

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载模型和设置随机种子
np.random.seed(42)
model = load_model('./CRISPR_MFH_best_model.h5')

# 编码函数

def Encoding(guide_seq, off_seq, dim=7):
    code_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0],
                 '-': [0, 0, 0, 0, 1]}
    direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '-': 1}
    tlen = 24
    guide_seq = "-" * (tlen - len(guide_seq)) + guide_seq.upper()
    off_seq = "-" * (tlen - len(off_seq)) + off_seq.upper()

    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    pair_code = []
    on_encoded_matrix = np.zeros((24, 5), dtype=np.float32)
    off_encoded_matrix = np.zeros((24, 5), dtype=np.float32)

    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]

        if gRNA_list[i] == '_':
            gRNA_list[i] = '-'

        if off_list[i] == '_':
            off_list[i] = '-'

        gRNA_base_code = code_dict[gRNA_list[i].upper()]
        DNA_based_code = code_dict[off_list[i].upper()]
        diff_code = np.bitwise_or(gRNA_base_code, DNA_based_code)

        if(dim==7):
            dir_code = np.zeros(2)
            if gRNA_list[i] != "-" and off_list[i] != "-" and direction_dict[gRNA_list[i]] != direction_dict[off_list[i]]:
                if direction_dict[gRNA_list[i]] > direction_dict[off_list[i]]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1

            pair_code.append(np.concatenate((diff_code, dir_code)))
        else:
            pair_code.append(diff_code)
        on_encoded_matrix[i] = code_dict[gRNA_list[i]]
        off_encoded_matrix[i] = code_dict[off_list[i]]

    pair_code_matrix = np.array(pair_code, dtype=np.float32).reshape(1, 1, 24, dim)

    return pair_code_matrix, on_encoded_matrix.reshape(1, 1, 24, 5), off_encoded_matrix.reshape(1, 1, 24, 5)

# 绘制矩形函数

def draw_rect(ax, i, y, y2):
    left, right, top, bottom = i + 0.5, i + 1.5, max(y), min(y)
    color_num = (1 - (y2 - bottom) / (top - bottom)) * 3
    if 2 < color_num <= 3:
        red, green, blue = 1, 1, (color_num - 2) * 4 / 5 + 0.15
    elif 1 < color_num <= 2:
        red, green, blue = 1, color_num - 1, 0.15
    elif 0 <= color_num <= 1:
        red, green, blue = color_num * 3 / 5 + 0.4, 0, 0.15
        red = np.exp(red - 1)
    verts = [(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=(red, green, blue), linewidth=0, alpha=0.5)
    ax.add_patch(patch)

# 绘制 F1 图像

def draw_f1(ax, m_change):
    occlude = m_change.mean(axis=0)
    x = np.arange(1, 24)
    y = occlude.values

    for i, y2 in enumerate(y):
        draw_rect(ax, i, y, y2)
    ax.plot([0.5, 23.5], [0, 0], color="grey", linestyle="--", linewidth=0.5)
    ax.plot(x, y, color="blue", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_ylim(min(occlude), max(occlude))
    ax.set_xlim(0.5, 23.5)
    # ax.set_xlabel("Base position", fontsize=FONTSIZE)
    # ax.set_ylabel("Substitution score", fontsize=FONTSIZE)
    ax.set_title("(a) The impact of base substitutions at various Positions", fontsize=FONTSIZE)

# 绘制 F2-F4 图像

def draw_f234(ax, m_change, num):
    fig_num = {0: "b", 1: "c", 2: "d", 3: "e", 4: "f"}
    base_name = {0: "A-A", 1: "C-C", 2: "G-G", 3: "T-T", 4: "indel"}
    occlude = m_change.iloc[num]
    c = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]
    x = np.arange(1, 24)
    y = occlude.values

    ax.plot([0.5, 23.5], [0, 0], color="grey", linestyle="--", linewidth=0.5)
    ax.bar(x, y, color=c[num], edgecolor="black")
    ax.set_xticks(x)
    ax.grid(axis="x", color="grey", linestyle="--", linewidth=0.5)
    ax.set_ylim(min(occlude), max(occlude))
    ax.set_xlim(0.5, 23.5)
    # ax.set_xlabel("Base position", fontsize=FONTSIZE)
    # ax.set_ylabel("Substitution score", fontsize=FONTSIZE)
    ax.set_title(f"({fig_num[num]}) The impact of positional substitutions on the predictive score for '{base_name[num]}' ", fontsize=FONTSIZE)

# 绘制热图


def draw_heatmap(ax, m_change):
    sns.heatmap(data=m_change,
                cmap=sns.diverging_palette(10, 220, sep=130, n=30, center="light"),
                linewidths=0.7,
                cbar_kws={},
                xticklabels=np.arange(1, 24),
                yticklabels=["A", "C", "G", "T"],
                square=True,
                ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONTSIZE)
    # ax.set_xlabel("Base", fontsize=FONTSIZE)
    # ax.set_ylabel("Base position", fontsize=FONTSIZE)
    ax.set_title("(g) The heat map of the off-target effect score changes due to base substitution at each position", fontsize=FONTSIZE)


import matplotlib.gridspec as gridspec

if __name__ == "__main__":
    time1 = time.time()

    FONTSIZE = 12
    plt.figure(dpi=300, figsize=(12, 8))
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
              "legend.fontsize": FONTSIZE,
              "axes.labelsize": FONTSIZE,
              "xtick.labelsize": FONTSIZE,
              "ytick.labelsize": FONTSIZE,
              "figure.titlesize": FONTSIZE}
    plt.rcParams.update(params)

    # 加载数据
    m_change = pd.read_csv("./pred_matrix_change.csv")

    # 定义GridSpec布局
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.5])

    # 创建子图
    fig = plt.figure(figsize=(11, 12), dpi=300)

    # 绘制 F1-F4 图像
    ax1 = fig.add_subplot(gs[0, 0])
    draw_f1(ax1, m_change)

    for i in range(5):
        ax = fig.add_subplot(gs[(i + 1) // 2, (i + 1) % 2])
        draw_f234(ax, m_change, i)

    # 绘制热力图，设置其跨越第4行的两个列
    heatmap_ax = fig.add_subplot(gs[3, :])
    draw_heatmap(heatmap_ax, m_change)

    plt.tight_layout()
    plt.savefig(fname="visual.png", format="png", bbox_inches="tight")
    plt.show()
    print(time.time() - time1)
