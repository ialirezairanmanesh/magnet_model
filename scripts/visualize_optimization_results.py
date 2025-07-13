import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_f1_scores_over_trials(results, output_dir):
    """Plot F1 scores over trials"""
    plt.figure(figsize=(10, 6))
    
    # Extract trial numbers and F1 scores
    trials = []
    f1_scores = []
    
    for trial in results['all_trials']:
        trials.append(trial['trial_number'])
        f1_scores.append(-trial['neg_f1'])  # Convert negative F1 to positive
    
    # Plot
    plt.plot(trials, f1_scores, 'o-', label='F1 Score per Trial')
    plt.plot(trials, np.maximum.accumulate(f1_scores), 'r--', label='Best F1 Score So Far')
    
    plt.xlabel('Trial Number')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores Over Optimization Trials')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_dir / 'f1_scores_over_trials.png')
    plt.close()

def plot_parameter_distributions(results, output_dir):
    """Plot distributions of key parameters for top trials"""
    # Get top 10 trials by F1 score
    top_trials = sorted(results['all_trials'], 
                       key=lambda x: -x['neg_f1'])[:10]
    
    # Parameters to plot
    params = ['embedding_dim', 'num_heads', 'num_layers', 'dropout']
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.ravel()
    
    for i, param in enumerate(params):
        values = [trial['metrics']['config'][param] for trial in top_trials]
        f1_scores = [-trial['neg_f1'] for trial in top_trials]
        
        # Scatter plot
        scatter = axs[i].scatter(values, f1_scores, c=f1_scores, cmap='viridis')
        axs[i].set_xlabel(param)
        axs[i].set_ylabel('F1 Score')
        axs[i].set_title(f'{param} vs F1 Score')
        axs[i].grid(True)
        
        # Add colorbar
        plt.colorbar(scatter, ax=axs[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png')
    plt.close()

def plot_metrics_comparison(results, output_dir):
    """Plot comparison of different metrics for top trials"""
    # Get top 10 trials
    top_trials = sorted(results['all_trials'], 
                       key=lambda x: -x['neg_f1'])[:10]
    
    # Extract metrics
    metrics = ['f1', 'accuracy', 'precision', 'recall']
    trial_numbers = [t['trial_number'] for t in top_trials]
    
    plt.figure(figsize=(12, 6))
    
    for metric in metrics:
        values = [t['metrics'][metric] for t in top_trials]
        plt.plot(trial_numbers, values, 'o-', label=metric.capitalize())
    
    plt.xlabel('Trial Number')
    plt.ylabel('Score')
    plt.title('Metrics Comparison for Top Trials')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / 'metrics_comparison.png')
    plt.close()

def main():
    # Setup paths
    base_dir = Path('/home/alireza/Documents/final_magnet/magnet_model')
    results_dir = base_dir / 'results/pirates_optimization'
    plots_dir = base_dir / 'docs/plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load results
    results_file = results_dir / 'pirates_results_20250421_011535.json'
    results = load_results(results_file)
    
    # Generate plots
    plot_f1_scores_over_trials(results, plots_dir)
    plot_parameter_distributions(results, plots_dir)
    plot_metrics_comparison(results, plots_dir)
    
    print(f"Plots have been generated in {plots_dir}")

if __name__ == "__main__":
    main() 