import matplotlib.pyplot as plt
import numpy as np

# Data for literature comparison
lit_models = ['MAGNET', 'Multimodal', 'Transformer']
lit_accuracies = [0.9724, 0.892, 0.958]
lit_f1_scores = [0.9823, None, None]
lit_aucs = [0.9932, None, None]

# Data for ML models comparison
ml_models = ['SVM', 'Random Forest', 'XGBoost', 'ANN', 'CNN', 'LSTM', 'MAGNET']
ml_datasets = ['CICAndMal2017', 'Malgenome', 'Malgenome', 'DREBIN', 'VX-Heaven', 'CICAndMal2017', 'DREBIN']
ml_accuracies = [0.985, 0.945, 0.958, 0.962, 0.966, 0.883, 0.972]
ml_precisions = [0.995, 0.940, 0.955, 0.965, 0.960, 0.875, 0.980]
ml_recalls = [0.996, 0.950, 0.960, 0.959, 0.970, 0.890, 0.985]
ml_f1_scores = [0.995, 0.945, 0.957, 0.962, 0.965, 0.882, 0.982]
ml_aucs = [0.985, 0.970, 0.975, 0.985, 0.980, 0.920, 0.993]

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