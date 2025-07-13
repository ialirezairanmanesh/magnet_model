import matplotlib.pyplot as plt
import numpy as np

# Data for literature comparison
lit_models = ['MAGNET', 'DeepImageDroid \cite{Obidiagha2024}', 'Graph-BERT \cite{White2023}']
lit_accuracies = [0.9724, 0.968, 0.962]
lit_f1_scores = [0.9823, 0.974, 0.968]
lit_aucs = [0.9932, 0.988, 0.985]

# Data for ML models comparison
ml_models = ['SVM \cite{AndroidMalwareSurvey}', 'Random Forest \cite{DeepLearningMalware}', 'XGBoost \cite{AndroidMalwareSurvey}', 'ANN \cite{DeepLearningMalware}', 'MAGNET']
ml_datasets = ['DREBIN', 'DREBIN', 'DREBIN', 'DREBIN', 'DREBIN']
ml_accuracies = [0.906, 0.935, 0.948, 0.962, 0.972]
ml_precisions = [0.915, 0.942, 0.953, 0.965, 0.980]
ml_recalls = [0.892, 0.928, 0.943, 0.959, 0.985]
ml_f1_scores = [0.903, 0.935, 0.948, 0.962, 0.982]
ml_aucs = [0.945, 0.967, 0.978, 0.985, 0.993]

# Create figure for literature comparison
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = np.arange(len(lit_models))

# Create bars
plt.bar(index, lit_accuracies, bar_width, label='Accuracy', color='blue', alpha=0.8)
plt.bar(index + bar_width, [v if v is not None else 0 for v in lit_f1_scores], bar_width, label='F1 Score', color='green', alpha=0.8)
plt.bar(index + 2*bar_width, [v if v is not None else 0 for v in lit_aucs], bar_width, label='AUC', color='red', alpha=0.8)

# Add labels and title
plt.xlabel('Models', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Comparison with Literature Methods', fontsize=14, pad=20)
plt.xticks(index + bar_width, lit_models, rotation=45, ha='right')

# Add value labels on top of bars
for i, v in enumerate(lit_accuracies):
    plt.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)
for i, v in enumerate(lit_f1_scores):
    if v is not None:
        plt.text(i + bar_width, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)
for i, v in enumerate(lit_aucs):
    if v is not None:
        plt.text(i + 2*bar_width, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)

plt.legend(prop={'size': 10})
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
plt.savefig('fig_literature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create figure for F1 Score and AUC comparison
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(ml_models))

# Create bars
plt.bar(index, ml_f1_scores, bar_width, label='F1 Score', color='blue', alpha=0.8)
plt.bar(index + bar_width, ml_aucs, bar_width, label='AUC', color='red', alpha=0.8)

# Add labels and title
plt.xlabel('Models', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Comparison of F1 Score and AUC for Different Models', fontsize=14, pad=20)
plt.xticks(index + bar_width/2, ml_models, rotation=45, ha='right')

# Add dataset labels
for i, d in enumerate(ml_datasets):
    plt.text(index[i] + bar_width/2, -0.02, d, rotation=45, ha='right', va='top', fontsize=8)

# Add value labels on top of bars
for i, v in enumerate(ml_f1_scores):
    plt.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)
for i, v in enumerate(ml_aucs):
    plt.text(i + bar_width, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)

plt.legend(prop={'size': 10})
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
plt.savefig('fig_baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create figure for Accuracy, Precision, and Recall
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = np.arange(len(ml_models))

# Create bars
plt.bar(index - bar_width, ml_accuracies, bar_width, label='Accuracy', color='green', alpha=0.8)
plt.bar(index, ml_precisions, bar_width, label='Precision', color='purple', alpha=0.8)
plt.bar(index + bar_width, ml_recalls, bar_width, label='Recall', color='orange', alpha=0.8)

# Add labels and title
plt.xlabel('Models', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Comparison of Accuracy, Precision, and Recall for Different Models', fontsize=14, pad=20)
plt.xticks(index, ml_models, rotation=45, ha='right')

# Add dataset labels
for i, d in enumerate(ml_datasets):
    plt.text(index[i], -0.02, d, rotation=45, ha='right', va='top', fontsize=8)

# Add value labels on top of bars
for i, v in enumerate(ml_accuracies):
    plt.text(i - bar_width, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)
for i, v in enumerate(ml_precisions):
    plt.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)
for i, v in enumerate(ml_recalls):
    plt.text(i + bar_width, v + 0.002, f'{v:.3f}', ha='center', fontsize=8)

plt.legend(prop={'size': 10})
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
plt.savefig('fig_baseline_metrics.png', dpi=300, bbox_inches='tight')
plt.close() 