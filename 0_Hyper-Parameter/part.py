import matplotlib.pyplot as plt

# New data
x_values = [1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300]
crispr_mfh = [0.7429, 0.7413, 0.7367, 0.7521, 0.7228, 0.7508, 0.7450, 0.7469, 0.7390, 0.7486, 0.7506, 0.7527]
lstm = [0.7067, 0.7271, 0.7134, 0.7071, 0.7148, 0.7260, 0.7049, 0.7096, 0.7102, 0.7178, 0.7081, 0.7002]

# Create a single plot
plt.figure(figsize=(5, 3))
plt.plot(x_values, crispr_mfh, marker='o', label="CRISPR-MFH" )
plt.plot(x_values, lstm, marker='s', label="LSTM" )



# Save and show the plot
plt.tight_layout()
plt.savefig('./input_size_pr_auc.png', bbox_inches='tight', dpi=300, format='png')  # Save the plot
plt.show()
