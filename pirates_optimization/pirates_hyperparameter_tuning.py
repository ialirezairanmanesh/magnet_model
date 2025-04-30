#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: Hyperparameter Optimization with Pirates Algorithm
=========================================================
This script performs advanced hyperparameter optimization for the MAGNET model
using the Pirates algorithm and provides detailed analysis and visualization for comparison
with other optimization methods like Optuna.

Features:
- Efficient hyperparameter search using Pirates optimization algorithm
- Advanced visualization of optimization process
- Comprehensive results storage for academic analysis
- Resource-aware optimization with configurable settings

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# Import Pirates algoritme
from pirates import Pirates

# ML libraries
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
import torch.serialization
# Add the safe globals for PyTorch Geometric
torch.serialization.add_safe_globals([DataEdgeAttr])

# Import model en data loading utilities
from magnet_model import MAGNET
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Stel paden in
BASE_DIR = Path('/home/alireza/Documents/final_magnet/magnet_model')
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results/pirates_optimization'
MODELS_DIR = BASE_DIR / 'models'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Zet willekeurige seeds voor reproduceerbaarheid
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# Check memory capacity - adjust search space accordingly
if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
    # Set resource constraints based on available memory
    LOW_RESOURCE = gpu_mem_gb < 4
else:
    # Assume CPU-only operation is resource constrained
    LOW_RESOURCE = True
    print("Running on CPU. Using low-resource mode.")

# Custom collate function voor PyTorch Geometric Data objects
def custom_collate(batch):
    tabular_data = torch.stack([item[0][0] for item in batch])
    graph_data = batch[0][0][1]  # Since graph data is shared across all samples
    seq_data = torch.stack([item[0][2] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return (tabular_data, graph_data, seq_data), targets

# Dataset voor multimodale data
class MultiModalDataset(Dataset):
    def __init__(self, X_tabular, graph_data, seq_data, y):
        self.X_tabular = X_tabular
        self.graph_data = graph_data
        self.seq_data = seq_data
        
        # Normalize labels to ensure they are 0 or 1
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        y = np.array(y)
        y = y.flatten()
        y = np.where(y < 0, 0, y)
        y = np.where(y > 1, 1, y)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X_tabular[idx], self.graph_data, self.seq_data[idx]), self.y[idx]

# Map de Pirates parameters naar de MAGNET model parameters
def get_config_from_position(position):
    """
    Converteert een positie in de Pirates search space naar een model configuratie
    
    Args:
        position: Een array met waarden tussen 0 en 1
        
    Returns:
        dict: Configuratie dictionary met hyperparameter waarden
    """
    # Definieer bounds en maak een mapping van elke 0-1 waarde naar de juiste hyperparameter range
    embedding_dims = [16, 32, 64, 128]
    num_heads_options = [2, 4, 8, 16]
    dim_feedforward_options = [64, 128, 256, 512]
    batch_sizes = [16, 32, 64, 128]
    
    # Map position waarden naar hyperparameters
    config = {
        'embedding_dim': embedding_dims[min(int(position[0] * len(embedding_dims)), len(embedding_dims)-1)],
        'num_heads': num_heads_options[min(int(position[1] * len(num_heads_options)), len(num_heads_options)-1)],
        'num_layers': min(int(position[2] * 4) + 1, 4),  # 1-4 layers
        'dim_feedforward': dim_feedforward_options[min(int(position[3] * len(dim_feedforward_options)), len(dim_feedforward_options)-1)],
        'dropout': position[4] * 0.5,  # 0-0.5 dropout
        'batch_size': batch_sizes[min(int(position[5] * len(batch_sizes)), len(batch_sizes)-1)],
        'learning_rate': 10 ** (position[6] * (np.log10(1e-2) - np.log10(1e-4)) + np.log10(1e-4)),  # 1e-4 tot 1e-2
        'weight_decay': 10 ** (position[7] * (np.log10(1e-1) - np.log10(1e-5)) + np.log10(1e-5)),  # 1e-5 tot 1e-1
        'num_epochs': 1 if '--test' in sys.argv else (3 if LOW_RESOURCE else 5)  # کاهش تعداد اپوک‌ها برای سرعت بیشتر
    }
    
    # Zorg ervoor dat embedding_dim deelbaar is door num_heads
    valid_heads = [h for h in num_heads_options if h <= config['embedding_dim'] and config['embedding_dim'] % h == 0]
    if valid_heads:
        config['num_heads'] = valid_heads[min(int(position[1] * len(valid_heads)), len(valid_heads)-1)]
    else:
        # Default to 2 if no valid divisors
        config['num_heads'] = 2
    
    return config

# Train- en evaluatiefunctie voor de MAGNET model
def train_and_evaluate_magnet(position, X_tabular_train, X_tabular_test, graph_data, seq_train, seq_test, y_train, y_test):
    """
    Train en evalueer het MAGNET model met de gegeven hyperparameters
    
    Args:
        position: Positie in de Pirates search space
        X_tabular_train, X_tabular_test: Tabulaire features
        graph_data: Graph data (gedeeld over alle samples)
        seq_train, seq_test: Sequentiële data
        y_train, y_test: Target labels
        
    Returns:
        tuple: (error, metrics)
            error: Negatieve F1 score (omdat Pirates minimaliseert)
            metrics: Dictionary met extra metrics
    """
    # Vertaal positie naar hyperparameter configuratie
    config = get_config_from_position(position)
    
    # Log de configuratie
    print(f"\nTesting configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Use a smaller subset of data in test mode
    data_percentage = 2 if '--test' in sys.argv else 50  # 50% در حالت معمولی برای تعادل بین سرعت و دقت
    
    if data_percentage < 100:
        # Sample a subset of data
        indices = np.random.choice(len(y_train), size=int(len(y_train) * data_percentage / 100), replace=False)
        X_tabular_train_subset = X_tabular_train[indices]
        seq_train_subset = seq_train[indices]
        y_train_subset = y_train[indices]
        
        val_indices = np.random.choice(len(y_test), size=int(len(y_test) * data_percentage / 100), replace=False)
        X_tabular_test_subset = X_tabular_test[val_indices]
        seq_test_subset = seq_test[val_indices]
        y_test_subset = y_test[val_indices]
        
        print(f"Using {data_percentage}% of data: {len(y_train_subset)}/{len(y_train)} train samples, {len(y_test_subset)}/{len(y_test)} test samples")
    else:
        X_tabular_train_subset = X_tabular_train
        seq_train_subset = seq_train
        y_train_subset = y_train
        X_tabular_test_subset = X_tabular_test
        seq_test_subset = seq_test
        y_test_subset = y_test
    
    # Bereid datasets voor
    train_dataset = MultiModalDataset(X_tabular_train_subset, graph_data, seq_train_subset, y_train_subset)
    test_dataset = MultiModalDataset(X_tabular_test_subset, graph_data, seq_test_subset, y_test_subset)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate)
    
    # Instantieer het MAGNET model
    model = MAGNET(
        tabular_dim=X_tabular_train_subset.size(1),
        graph_node_dim=graph_data.x.size(1),
        graph_edge_dim=graph_data.edge_attr.size(1) if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else 0,
        seq_vocab_size=1000,  # Vocabulary size
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        seq_max_len=seq_train_subset.size(1)
    ).to(device)
    
    # Bereken class weights gebaseerd op genormaliseerde labels
    y_train_tensor = train_dataset.y
    class_counts = torch.bincount(y_train_tensor)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Definieer loss functie en optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=config['learning_rate'] * 0.01
    )
    
    # Training loop with progress bar
    best_val_f1 = 0.0
    all_epoch_metrics = []
    
    print("\nTraining progress:")
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Train)")
        for (tabular, graph, seq), targets in train_progress:
            tabular, graph, seq, targets = tabular.to(device), graph.to(device), seq.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _, _, _ = model(tabular, graph, seq)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        
        # Evaluation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Use tqdm for progress bar
        val_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Val)")
        with torch.no_grad():
            for (tabular, graph, seq), targets in val_progress:
                tabular, graph, seq, targets = tabular.to(device), graph.to(device), seq.to(device), targets.to(device)
                outputs, _, _, _ = model(tabular, graph, seq)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                val_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate metrics
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        
        # Save best F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
        # Store all metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss / len(test_loader) if len(test_loader) > 0 else float('inf'),
            'f1': val_f1,
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall
        }
        all_epoch_metrics.append(epoch_metrics)
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={epoch_metrics['val_loss']:.4f}")
        print(f"Metrics: F1={val_f1:.4f}, Accuracy={val_accuracy:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}")
        
        # Update scheduler
        scheduler.step()
    
    # Find best epoch metrics
    best_epoch = max(all_epoch_metrics, key=lambda x: x['f1'])
    
    # Return negative F1 score for minimization and metrics
    metrics = {
        'f1': best_epoch['f1'],
        'accuracy': best_epoch['accuracy'],
        'precision': best_epoch['precision'],
        'recall': best_epoch['recall'],
        'config': config,
        'epochs_data': all_epoch_metrics
    }
    
    return -best_epoch['f1'], metrics

# Helper class for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__class__') and 'Data' in obj.__class__.__name__:
            return str(type(obj))  # Just return the type name instead of trying to serialize the Data object
        return super(NumpyEncoder, self).default(obj)

# Objective function class for Pirates optimization
class HyperparameterOptimization:
    def __init__(self, dimensions=8, resume=True):
        self.dimensions = dimensions
        self.best_metrics = None
        self.best_error = float('inf')
        self.data_loaded = False
        self.all_trials = []  # Store all trial results
        self.load_data()
        
        # Try to resume from previous checkpoint
        if resume:
            self.load_progress()
    
    def load_data(self):
        """
        Load data from processed_data directory
        """
        try:
            # Path to processed data
            processed_data_dir = Path('/home/alireza/Documents/final_magnet/magnet_model/processed_data')
            
            print("Loading data from processed_data directory...")
            # Load all the necessary data files
            self.X_tabular_train = torch.load(processed_data_dir / 'X_tabular_train.pt', weights_only=False)
            self.X_tabular_test = torch.load(processed_data_dir / 'X_tabular_test.pt', weights_only=False)
            self.graph_data = torch.load(processed_data_dir / 'graph_data_processed.pt', weights_only=False)
            self.seq_train = torch.load(processed_data_dir / 'seq_train.pt', weights_only=False)
            self.seq_test = torch.load(processed_data_dir / 'seq_test.pt', weights_only=False)
            self.y_train = torch.load(processed_data_dir / 'y_train.pt', weights_only=False)
            self.y_test = torch.load(processed_data_dir / 'y_test.pt', weights_only=False)
            
            # Move graph data to device
            self.graph_data = self.graph_data.to(device)
            
            print(f"Data loaded successfully: {self.X_tabular_train.shape[0]} training samples, {self.X_tabular_test.shape[0]} test samples")
            self.data_loaded = True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def load_progress(self):
        """Load previous optimization progress if exists"""
        progress_file = RESULTS_DIR / "optimization_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    
                if 'best_metrics' in progress_data and progress_data['best_metrics']:
                    self.best_metrics = progress_data['best_metrics']
                    self.best_error = self.best_metrics.get('error', float('inf'))
                    print(f"Loaded best metrics with F1 score: {-self.best_error:.4f}")
                
                if 'all_trials' in progress_data and progress_data['all_trials']:
                    self.all_trials = progress_data['all_trials']
                    print(f"Resuming from {len(self.all_trials)} previous trials")
                
                return True
            except Exception as e:
                print(f"Could not load previous progress: {str(e)}")
        
        return False
        
    def func(self, position):
        """
        Objective function for Pirates optimization.
        This function takes a position in the search space and returns the negative F1 score.
        """
        # Convert position to hyperparameters
        config = get_config_from_position(position)
        
        # Add data to config
        config['X_tabular_train'] = self.X_tabular_train
        config['X_tabular_test'] = self.X_tabular_test
        config['graph_data'] = self.graph_data
        config['seq_train'] = self.seq_train
        config['seq_test'] = self.seq_test
        config['y_train'] = self.y_train
        config['y_test'] = self.y_test
        
        # Train and evaluate the model
        trial_number = len(self.all_trials) + 1
        print(f"\n{'='*80}")
        print(f"Trial {trial_number} - Starting evaluation")
        print(f"{'='*80}")
        
        neg_f1, metrics = train_and_evaluate_magnet(position, self.X_tabular_train, self.X_tabular_test, self.graph_data, self.seq_train, self.seq_test, self.y_train, self.y_test)
        
        # Store the current trial
        trial_data = {
            'trial_number': trial_number,
            'position': position.tolist(),
            'neg_f1': neg_f1,
            'metrics': metrics
        }
        self.all_trials.append(trial_data)
        
        # Store metrics if this is the best result so far
        if neg_f1 < self.best_error:
            print(f"\n🌟 New best result! F1 score: {-neg_f1:.4f}")
            self.best_error = neg_f1
            metrics['error'] = neg_f1
            metrics['position'] = position.tolist()
            metrics['config'] = config
            self.best_metrics = metrics
            
        # Save progress after each trial
        self.save_progress()
            
        return neg_f1  # Return negative F1 score for minimization
    
    def save_progress(self):
        """Save current progress to avoid data loss if process is interrupted"""
        progress_file = RESULTS_DIR / f"optimization_progress.json"
        try:
            # Remove data objects that can't be serialized
            best_metrics_clean = None
            if self.best_metrics:
                best_metrics_clean = self.best_metrics.copy()
                if 'config' in best_metrics_clean:
                    config_clean = {k: v for k, v in best_metrics_clean['config'].items() 
                                 if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 
                                             'seq_train', 'seq_test', 'y_train', 'y_test']}
                    best_metrics_clean['config'] = config_clean
            
            # Clean trial data similarly
            all_trials_clean = []
            for trial in self.all_trials:
                trial_clean = trial.copy()
                if 'metrics' in trial_clean and 'config' in trial_clean['metrics']:
                    config_clean = {k: v for k, v in trial_clean['metrics']['config'].items()
                                 if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 
                                             'seq_train', 'seq_test', 'y_train', 'y_test']}
                    trial_clean['metrics']['config'] = config_clean
                all_trials_clean.append(trial_clean)
            
            progress_data = {
                'best_metrics': best_metrics_clean,
                'all_trials': all_trials_clean
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, cls=NumpyEncoder, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {str(e)}")

# Create visualizations from optimization results
def create_visualizations(optimization_results, output_dir):
    """
    Create and save optimization visualizations
    
    Args:
        optimization_results: Dictionary with optimization results
        output_dir: Directory to save plots
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    all_trials = optimization_results.get('all_trials', [])
    best_metrics = optimization_results.get('best_metrics', {})
    
    if not all_trials:
        print("No trials data available for visualization")
        return
    
    # Extract trial data
    trial_numbers = [trial['trial_number'] for trial in all_trials]
    f1_scores = [-trial['neg_f1'] for trial in all_trials]
    best_f1_so_far = [max(f1_scores[:i+1]) for i in range(len(f1_scores))]
    
    # 1. Optimization history plot
    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, f1_scores, 'o-', label='Trial F1 Score')
    plt.plot(trial_numbers, best_f1_so_far, 's-', label='Best F1 So Far')
    plt.xlabel('Trial Number')
    plt.ylabel('F1 Score')
    plt.title('Pirates Optimization History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
    plt.close()
    
    # 2. Hyperparameter distribution for top trials
    top_k = min(10, len(all_trials))
    top_trials = sorted(all_trials, key=lambda x: -x['neg_f1'])[:top_k]
    
    # Get all parameters from first trial
    if top_trials and 'metrics' in top_trials[0] and 'config' in top_trials[0]['metrics']:
        config_keys = [k for k in top_trials[0]['metrics']['config'].keys() 
                      if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 
                                  'seq_train', 'seq_test', 'y_train', 'y_test']]
        
        # Plot distribution of each parameter
        n_params = len(config_keys)
        if n_params > 0:
            fig, axs = plt.subplots(n_params, 1, figsize=(10, 4 * n_params))
            if n_params == 1:
                axs = [axs]  # Convert to list for consistent indexing
                
            for i, param in enumerate(config_keys):
                param_values = [trial['metrics']['config'][param] for trial in top_trials if param in trial['metrics']['config']]
                if param_values:
                    if isinstance(param_values[0], (int, float)):
                        # For numeric parameters
                        axs[i].barh(range(len(param_values)), param_values)
                        axs[i].set_yticks(range(len(param_values)))
                        axs[i].set_yticklabels([f"Trial {trial['trial_number']} (F1={-trial['neg_f1']:.4f})" 
                                            for trial in top_trials])
                    else:
                        # For categorical parameters
                        value_counts = {}
                        for val in param_values:
                            value_counts[val] = value_counts.get(val, 0) + 1
                        axs[i].bar(value_counts.keys(), value_counts.values())
                        
                    axs[i].set_title(f'Distribution of {param} in Top {top_k} Trials')
                    axs[i].set_xlabel(param)
                    
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'param_distributions.png'))
            plt.close()
    
    # 3. Training curves for the best trial
    if best_metrics and 'epochs_data' in best_metrics:
        epochs_data = best_metrics['epochs_data']
        if epochs_data:
            epochs = [data['epoch'] for data in epochs_data]
            metrics_to_plot = ['f1', 'accuracy', 'precision', 'recall']
            
            plt.figure(figsize=(12, 6))
            for metric in metrics_to_plot:
                if metric in epochs_data[0]:
                    values = [data[metric] for data in epochs_data]
                    plt.plot(epochs, values, 'o-', label=metric.capitalize())
                    
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Training Metrics for Best Trial')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'best_trial_metrics.png'))
            plt.close()
            
            # Loss curves
            plt.figure(figsize=(12, 6))
            train_losses = [data['train_loss'] for data in epochs_data]
            val_losses = [data['val_loss'] for data in epochs_data]
            plt.plot(epochs, train_losses, 'o-', label='Training Loss')
            plt.plot(epochs, val_losses, 's-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curves for Best Trial')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'best_trial_losses.png'))
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main execution function for Pirates hyperparameter optimization"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"pirates_results_{timestamp}.json"
    
    # Add command line argument for resuming
    parser = argparse.ArgumentParser(description='MAGNET Hyperparameter Optimization with Pirates')
    parser.add_argument('--resume', action='store_true', help='Resume from previous checkpoint')
    parser.add_argument('--no-resume', action='store_false', dest='resume', help='Start fresh optimization')
    parser.add_argument('--test', action='store_true', help='Run in test mode with minimal parameters')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    
    # Print header
    print(f"\n{'='*80}")
    print(f"MAGNET Model Hyperparameter Optimization with Pirates Algorithm")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {results_file}")
    if args.resume:
        print(f"Attempting to resume from previous checkpoint")
    else:
        print(f"Starting fresh optimization (resume disabled)")
    print(f"{'='*80}")
    
    # Configure parameters based on resources and test mode
    if args.test:
        # Test mode with minimal parameters
        num_ships = 2
        max_iter = 2
        print("Running in TEST mode with minimal parameters")
    elif LOW_RESOURCE:
        num_ships = 10
        max_iter = 20
        print("Running in low-resource mode with reduced parameters")
    else:
        num_ships = 20
        max_iter = 40
        print("Running in standard mode with full parameters")
    
    # Initialize optimization function
    opt_func = HyperparameterOptimization(dimensions=8, resume=args.resume)
    
    # Adjust max_iter based on previous trials if resuming
    completed_trials = len(opt_func.all_trials)
    if args.resume and completed_trials > 0:
        # Reduce remaining iterations based on completed trials
        # For example, if each ship produces one trial, and we have 10 ships:
        completed_iterations = completed_trials // num_ships
        remaining_iterations = max(1, max_iter - completed_iterations)
        print(f"Already completed {completed_trials} trials ({completed_iterations} iterations)")
        print(f"Remaining iterations: {remaining_iterations}")
        max_iter = remaining_iterations
    
    # Configure and start Pirates optimization
    pirates = Pirates(
        func=opt_func,  # Pass the optimization object, not just the function
        fmin=(0, 0, 0, 0, 0, 0, 0, 0),  # Minimum values (all 0)
        fmax=(1, 1, 1, 1, 1, 1, 1, 1),  # Maximum values (all 1)
        hr=0.2,                       # Helper ratio
        ms=5,                         # Map size
        max_r=1,                      # Maximum radius
        num_ships=num_ships,          # Number of ships (particles)
        dimensions=8,                 # Number of dimensions (hyperparameters)
        max_iter=max_iter,            # Maximum number of iterations
        max_wind=0.5,                 # Maximum wind speed
        c={},                         # Constant weights
        top_ships=5,                  # Number of top ships
        dynamic_sails=True,           # Dynamic sails
        iteration_plots=False,        # Plot curves during iterations
        quiet=False,                  # Quiet mode
        sailing_radius=0.3,           # Initial sailing radius
        plundering_radius=0.1         # Initial plundering radius
    )
    
    # Run the optimization with progress tracking
    print("\nStarting Pirates optimization...")
    print(f"Configuration: {num_ships} ships, {max_iter} iterations")
    
    try:
        best_position, best_cost, best_metrics = pirates.search()
        
        # Get the best result from the optimization function
        best_metrics = opt_func.best_metrics
        
        if best_metrics:
            # Convert to f1_score for reporting (remove negative sign)
            best_f1 = -best_metrics['error'] if 'error' in best_metrics else 0
            best_config = best_metrics['config'] if 'config' in best_metrics else {}
            
            # Print results
            print(f"\n{'='*80}")
            print(f"Optimization completed in {(time.time() - start_time)/60:.2f} minutes")
            print(f"{'='*80}")
            print("\nBest hyperparameters:")
            for k, v in best_config.items():
                if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 'seq_train', 'seq_test', 'y_train', 'y_test']:
                    print(f"  {k}: {v}")
            print(f"Best F1 Score: {best_f1:.4f}")
            
            # Save best hyperparameters
            saveable_config = {k: str(v) if isinstance(v, torch.Tensor) else v for k, v in best_config.items() 
                             if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 'seq_train', 'seq_test', 'y_train', 'y_test']}
            
            # Prepare results
            optimization_results = {
                'best_params': saveable_config,
                'best_f1_score': best_f1,
                'best_position': best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                'timestamp': timestamp,
                'duration_minutes': (time.time() - start_time)/60,
                'all_trials': opt_func.all_trials
            }
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, cls=NumpyEncoder, indent=2)
            
            print(f"Results saved to {results_file}")
            
            # Create visualizations
            print("\nGenerating visualizations...")
            visualization_dir = PLOTS_DIR / f"pirates_{timestamp}"
            create_visualizations(optimization_results, visualization_dir)
            
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No valid results found")
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        # Save progress even when interrupted
        if hasattr(opt_func, 'all_trials') and opt_func.all_trials:
            interrupted_results = {
                'status': 'interrupted',
                'timestamp': timestamp,
                'duration_minutes': (time.time() - start_time)/60,
                'best_metrics': opt_func.best_metrics,
                'all_trials': opt_func.all_trials
            }
            
            interrupted_file = RESULTS_DIR / f"pirates_interrupted_{timestamp}.json"
            with open(interrupted_file, 'w') as f:
                json.dump(interrupted_results, f, cls=NumpyEncoder, indent=2)
            
            print(f"Partial results saved to {interrupted_file}")
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 