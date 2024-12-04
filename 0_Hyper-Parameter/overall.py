import matplotlib.pyplot as plt

# Data for the first plot
batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
crispr_mfh1 = [0.6336, 0.7218, 0.7253, 0.7128, 0.7388, 0.7447, 0.751, 0.7571, 0.7429, 0.7292, 0.6932, 0.6002, 0.452]
lstm1 = [0.6155, 0.6608, 0.6578, 0.6796, 0.6865, 0.701, 0.7194, 0.71, 0.7134, 0.7113, 0.6979, 0.6291, 0.4944]

# Data for the second plot
Random_Seeds = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
crispr_mfh2 = [0.3471, 0.3044, 0.3858, 0.3845, 0.332, 0.2578, 0.343, 0.3404, 0.2637, 0.3683, 0.3233]
lstm2 = [0.2311, 0.2108, 0.274, 0.2534, 0.2178, 0.2481, 0.2172, 0.2812, 0.2229, 0.1932, 0.2844]

# Plotting both graphs side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plotting the first graph
ax1.plot(batch_sizes, crispr_mfh1, marker='o', label="CRISPR-MFH")
ax1.plot(batch_sizes, lstm1, marker='s', label="LSTM")
ax1.set_xscale('log')
ax1.set_xlabel('Batch Size', fontsize=14)
ax1.set_ylabel('PR_AUC', fontsize=14)
ax1.set_title('(a) The impact of batch size on PR_AUC values', fontsize=16)
ax1.legend()

# Plotting the second graph
ax2.plot(Random_Seeds, crispr_mfh2, marker='o', label="CRISPR-MFH")
ax2.plot(Random_Seeds, lstm2, marker='s', label="LSTM")
ax2.set_xlabel('Random Seed', fontsize=14)
ax2.set_ylabel('PR_AUC', fontsize=14)
ax2.set_title('(b) The impact of random seed on PR_AUC values', fontsize=16)
ax2.legend()

# Show plots
plt.tight_layout()
plt.savefig('./hb.png', bbox_inches='tight', dpi=300, format='png')  # 保存热力图
plt.show()