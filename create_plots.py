import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set font
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data with English method names
data = {
    'Method': [
        'MAGNET', 'DREBIN', 'LOF', 'PIKADROID',
        'CrossMalDroid', 'DroidAPIMiner', 'Multimodal', 'Transformer'
    ],
    'Accuracy': [97.24, 92.3, 94.1, 96.8, 95.2, 89.7, 89.2, 95.8],
    'F1 Score': [0.9823, 0.933, 0.918, 0.974, 0.952, 0.891, None, None],
    'AUC': [0.9932, 0.955, 0.981, 0.988, 0.976, 0.927, None, None]
}
df = pd.DataFrame(data)

plt.rcParams['font.size'] = 13

# Figure 1: Accuracy comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Method'], df['Accuracy'], color='royalblue', alpha=0.7)
plt.xticks(rotation=30, ha='right')
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy Comparison of Android Malware Detection Methods', fontsize=15, pad=20)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('images/fig_literature_comparison_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: F1 Score and AUC comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(df['Method']))
width = 0.35

bars1 = plt.bar(x - width/2, df['F1 Score'], width, label='F1 Score', color='mediumseagreen', alpha=0.7)
bars2 = plt.bar(x + width/2, df['AUC'], width, label='AUC', color='tomato', alpha=0.7)

plt.xlabel('Method', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('F1 Score and AUC Comparison of Android Malware Detection Methods', fontsize=15, pad=20)
plt.xticks(x, df['Method'], rotation=30, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('images/fig_literature_comparison_metrics.png', dpi=300, bbox_inches='tight')
plt.close() 