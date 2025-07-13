import matplotlib.pyplot as plt
import numpy as np

# Data from overall_comparison table
stages = ['Optimization\n(Validation)', 'Optuna\n(Validation)', 'Training\n(100% Data)', 
          'Cross-Validation', 'Test Set']
f1_scores = [0.9767, 0.9684, 0.9805, 0.9818, 0.9823]
accuracies = [0.9628, 0.9513, None, 0.9722, 0.9724]
aucs = [None, 0.9836, 0.9931, 0.9932, 0.9932]

# Create figure
plt.figure(figsize=(12, 6))

# Set width of bars
bar_width = 0.25
index = np.arange(len(stages))

# Create bars
plt.bar(index - bar_width, f1_scores, bar_width, label='F1 Score', color='blue', alpha=0.8)
plt.bar(index, [x if x is not None else 0 for x in accuracies], bar_width, label='Accuracy', color='green', alpha=0.8)
plt.bar(index + bar_width, [x if x is not None else 0 for x in aucs], bar_width, label='AUC', color='red', alpha=0.8)

# Add labels and title
plt.xlabel('Stages', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Overall Performance of MAGNET Model in Different Stages', fontsize=14, pad=20)
plt.xticks(index, stages, fontsize=10, rotation=45, ha='right')
plt.legend(prop={'size': 10})

# Add value labels on top of bars
for i, v in enumerate(f1_scores):
    plt.text(i - bar_width, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)
for i, v in enumerate(accuracies):
    if v is not None:
        plt.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)
for i, v in enumerate(aucs):
    if v is not None:
        plt.text(i + bar_width, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('fig_overall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()