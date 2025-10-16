import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib as mpl

# Set font for Persian text
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Data from cv_results table
categories = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
f1_scores = [0.9858, 0.9846, 0.9839, 0.9742, 0.9808]
accuracies = [0.9785, 0.9763, 0.9752, 0.9601, 0.9709]
aucs = [0.9950, 0.9955, 0.9945, 0.9861, 0.9946]

# Bar chart settings
bar_width = 0.25
index = np.arange(len(categories))

# Create the first figure (metrics comparison)
plt.figure(figsize=(10, 6))
plt.bar(index, f1_scores, bar_width, label='F1 Score', color='b')
plt.bar(index + bar_width, accuracies, bar_width, label='Accuracy', color='g')
plt.bar(index + 2 * bar_width, aucs, bar_width, label='AUC', color='r')

plt.xlabel('Folds', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Comparison of Metrics in 5-Fold Cross-Validation of MAGNET Model', fontsize=14, pad=20)
plt.xticks(index + bar_width, categories, fontsize=10)
plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig('fig_cv_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# Create the second figure (loss plot)
losses = [0.0786, 0.0735, 0.0839, 0.1199, 0.0864]

plt.figure(figsize=(8, 5))
plt.plot(categories, losses, marker='o', color='m', label='Loss')
plt.xlabel('Folds', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Changes in 5-Fold Cross-Validation of MAGNET Model', fontsize=14, pad=20)
plt.xticks(categories, fontsize=10)
plt.legend(prop={'size': 10})
plt.grid(True)
plt.tight_layout()
plt.savefig('fig_cv_loss.png', dpi=300, bbox_inches='tight')
plt.close() 