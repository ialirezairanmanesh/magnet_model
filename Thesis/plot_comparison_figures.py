import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the images directory exists
os.makedirs('images', exist_ok=True)

# Figure 1: Accuracy comparison
methods = ['Multimodal', 'Transformer-based', 'MAGNET']
accuracies = [89.2, 95.8, 97.24]
plt.figure(figsize=(8, 5))
bars = plt.bar(methods, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of MAGNET and Baseline Methods')
plt.ylim(0, 100)
plt.grid(True, axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/fig_literature_comparison.png', dpi=300)
plt.close()

# Figure 2: Metrics comparison
models = ['SVM', 'Random Forest', 'XGBoost', 'ANN', 'MAGNET']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
data = np.array([
    [0.906, 0.915, 0.892, 0.903, 0.945],
    [0.935, 0.942, 0.928, 0.935, 0.967],
    [0.948, 0.953, 0.943, 0.948, 0.978],
    [0.962, 0.965, 0.959, 0.962, 0.985],
    [0.972, 0.980, 0.985, 0.982, 0.993]
])
x = np.arange(len(models))
width = 0.15
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i*width, data[:, i], width, label=metric)
plt.xlabel('Models')
plt.ylabel('Value')
plt.title('Comparison of Different Metrics for MAGNET and Classical ML Models')
plt.xticks(x + width*2, models)
plt.ylim(0.85, 1.05)
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/fig_baseline_metrics_comparison.png', dpi=300)
plt.close() 