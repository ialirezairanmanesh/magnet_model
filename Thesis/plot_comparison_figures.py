import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the images directory exists
os.makedirs('images', exist_ok=True)

# Figure 1: Accuracy comparison
methods = ['DeepImageDroid \cite{Obidiagha2024}', 'Graph-BERT \cite{White2023}', 'DREBIN \cite{Drebin}', 'LOF \cite{Milosevic2017}', 'PIKADROID \cite{Pendlebury2020}', 'CrossMalDroid \cite{Martin2021}', 'DroidAPIMiner \cite{Aafer2013}', 'MAGNET']
accuracies = [96.8, 96.2, 92.3, 94.1, 96.8, 95.2, 89.7, 97.24]
plt.figure(figsize=(12, 6))
bars = plt.bar(methods, accuracies, color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'])
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of MAGNET and Literature Methods')
plt.ylim(85, 100)
plt.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/fig_literature_comparison_accuracy.png', dpi=300)
plt.close()

# Figure 2: Metrics comparison
models = ['SVM \cite{AndroidMalwareSurvey}', 'Random Forest \cite{DeepLearningMalware}', 'XGBoost \cite{AndroidMalwareSurvey}', 'ANN \cite{DeepLearningMalware}', 'MAGNET']
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

# Figure 3: F1 Score and AUC comparison
methods = ['DeepImageDroid \cite{Obidiagha2024}', 'Graph-BERT \cite{White2023}', 'DREBIN \cite{Drebin}', 'LOF \cite{Milosevic2017}', 'PIKADROID \cite{Pendlebury2020}', 'CrossMalDroid \cite{Martin2021}', 'DroidAPIMiner \cite{Aafer2013}', 'MAGNET']
f1_scores = [0.974, 0.968, 0.933, 0.918, 0.974, 0.952, 0.891, 0.9823]
aucs = [0.988, 0.985, 0.955, 0.981, 0.988, 0.976, 0.927, 0.9932]

plt.figure(figsize=(12, 6))
x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, f1_scores, width, label='F1 Score', color='#3498db')
plt.bar(x + width/2, aucs, width, label='AUC', color='#e74c3c')

plt.xlabel('Methods')
plt.ylabel('Value')
plt.title('Comparison of F1 Score and AUC for Different Methods')
plt.xticks(x, methods, rotation=45, ha='right')
plt.ylim(0.85, 1.0)
plt.legend()
plt.grid(True, axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(f1_scores):
    plt.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
for i, v in enumerate(aucs):
    plt.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('images/fig_literature_comparison_metrics.png', dpi=300)
plt.close() 