#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: Advanced Hyperparameter Optimization
====================================================
This script performs comprehensive hyperparameter optimization for the MAGNET model
using Optuna and provides detailed analysis and visualization for academic publication.

Features:
- Efficient hyperparameter search using Bayesian optimization
- Supports various optimizers, schedulers, and model architectures
- Cross-validation for more reliable performance estimates
- Resource-aware optimization to work within hardware constraints
- Comprehensive visualization and statistical analysis for publication

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import optuna
from optuna.visualization import (
    plot_optimization_history, plot_param_importances,
    plot_contour, plot_slice, plot_parallel_coordinate
)

# Safe imports for PyTorch Geometric data
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
import torch.serialization
torch.serialization.add_safe_globals([DataEdgeAttr])

# Import your model and data loading utils
from magnet_model import MAGNET
from create_dataloaders import MultiModalDataset, custom_collate
from data_extraction import load_processed_data

# Configure filesystem for plots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

# Set paths
RESULTS_DIR = Path("results/hyperparameter_optimization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Check available device and set it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# Function to define the search space based on available resources
def get_search_space(trial, low_resource=False):
    """
    Define search space for hyperparameters, adjusting based on available resources
    
    Args:
        trial: Optuna trial object
        low_resource: Whether to use a more constrained search space
        
    Returns:
        dict: Configuration dictionary with hyperparameter values
    """
    if low_resource:
        # More constrained search space for limited hardware
        config = {
            # Model architecture
            'embedding_dim': trial.suggest_categorical('embedding_dim', [16, 32, 64]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dim_feedforward': trial.suggest_categorical('dim_feedforward', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3, step=0.05),
            
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
            
            # Fixed parameters for low-resource setting
            'num_epochs': 10,  # Reduced epochs for quicker trials
            'patience': 5,     # Early stopping patience
            'data_percentage': 50  # Use 50% of data for faster iteration
        }
    else:
        # Fuller search space for more capable hardware
        config = {
            # Model architecture
            'embedding_dim': trial.suggest_categorical('embedding_dim', [32, 64, 128]),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            
            # Extra parameters to tune in higher-resource setting
            'num_epochs': 20,  # More epochs for more thorough training
            'patience': 7,     # Early stopping patience
            'data_percentage': 80  # Use 80% of data for better generalization
        }
    
    # Add optimizer selection
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
    config['optimizer'] = optimizer_name
    
    # Add learning rate scheduler selection
    scheduler_name = trial.suggest_categorical('scheduler', 
                                             ['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'OneCycleLR'])
    config['scheduler'] = scheduler_name
    
    # Ensure embedding_dim is divisible by num_heads
    if config['embedding_dim'] % config['num_heads'] != 0:
        # Adjust num_heads to be a divisor of embedding_dim
        valid_heads = [h for h in [2, 4, 8, 16] if config['embedding_dim'] % h == 0]
        if valid_heads:
            config['num_heads'] = trial.suggest_categorical('num_heads_adjusted', valid_heads)
        else:
            # If no valid divisors in our list, default to 2
            config['num_heads'] = 2
    
    return config

# Define function to get optimizer
def get_optimizer(optimizer_name, model_parameters, lr, weight_decay):
    """
    Create optimizer based on name and parameters
    
    Args:
        optimizer_name: Name of the optimizer to use
        model_parameters: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# Define function to get learning rate scheduler
def get_scheduler(scheduler_name, optimizer, config, steps_per_epoch=None):
    """
    Create learning rate scheduler based on name and parameters
    
    Args:
        scheduler_name: Name of the scheduler to use
        optimizer: The optimizer to schedule
        config: Configuration parameters
        steps_per_epoch: Number of batches per epoch (for certain schedulers)
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured scheduler
    """
    if scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=config['learning_rate'] * 0.01
        )
    elif scheduler_name == 'OneCycleLR':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for OneCycleLR")
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config['learning_rate'] * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=config['num_epochs'],
            anneal_strategy='cos'
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

# Training function with cross-validation for hyperparameter optimization
def train_and_evaluate_with_cv(trial, X_tabular, graph_data, seq_data, y, 
                             n_splits=3, low_resource=LOW_RESOURCE):
    """
    Train and evaluate MAGNET model with cross-validation
    
    Args:
        trial: Optuna trial object
        X_tabular: Tabular features
        graph_data: Graph data (shared across all samples)
        seq_data: Sequential data
        y: Target labels
        n_splits: Number of CV folds
        low_resource: Whether to use constrained hyperparameter space
        
    Returns:
        float: Mean F1 score across folds
    """
    # Get hyperparameter configuration
    config = get_search_space(trial, low_resource=low_resource)
    print(f"\nTrial {trial.number}: Testing configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Define cross-validation folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # If using a percentage of data, sample now
    if config['data_percentage'] < 100:
        indices = np.arange(len(y))
        sample_size = int(len(indices) * config['data_percentage'] / 100)
        np.random.shuffle(indices)
        selected_indices = indices[:sample_size]
        X_tabular_sampled = X_tabular[selected_indices]
        seq_data_sampled = seq_data[selected_indices]
        y_sampled = y[selected_indices]
        print(f"Using {config['data_percentage']}% of data: {len(y_sampled)}/{len(y)} samples")
    else:
        X_tabular_sampled = X_tabular
        seq_data_sampled = seq_data
        y_sampled = y
    
    # Metrics to collect across folds
    fold_metrics = []
    pruned = False
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tabular_sampled)):
        print(f"\nTraining fold {fold+1}/{n_splits}")
        
        # Create train/val datasets
        X_tabular_train = X_tabular_sampled[train_idx]
        X_tabular_val = X_tabular_sampled[val_idx]
        seq_train = seq_data_sampled[train_idx]
        seq_val = seq_data_sampled[val_idx]
        y_train = y_sampled[train_idx]
        y_val = y_sampled[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = MultiModalDataset(X_tabular_train, graph_data, seq_train, y_train)
        val_dataset = MultiModalDataset(X_tabular_val, graph_data, seq_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=custom_collate
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            collate_fn=custom_collate
        )
        
        # Initialize model with current hyperparameters
        model = MAGNET(
            tabular_dim=X_tabular_train.shape[1],
            graph_node_dim=graph_data.x.shape[1],
            graph_edge_dim=graph_data.edge_attr.shape[1] if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else 0,
            seq_vocab_size=1000,  # Adjust based on your dataset
            seq_max_len=seq_train.shape[1],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        ).to(device)
        
        # Calculate class weights for balanced training
        y_train_np = y_train if not isinstance(y_train, torch.Tensor) else y_train.numpy()
        class_counts = np.bincount(y_train_np.astype(int))
        class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # Define criterion, optimizer and scheduler
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = get_optimizer(
            config['optimizer'], 
            model.parameters(), 
            config['learning_rate'],
            config['weight_decay']
        )
        
        # Get scheduler (OneCycleLR needs steps_per_epoch)
        if config['scheduler'] == 'OneCycleLR':
            scheduler = get_scheduler(
                config['scheduler'], optimizer, config, 
                steps_per_epoch=len(train_loader)
            )
        else:
            scheduler = get_scheduler(config['scheduler'], optimizer, config)
        
        # Training loop
        best_val_f1 = 0.0
        early_stop_count = 0
        epoch_metrics = []
        
        for epoch in range(config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Train)") as pbar:
                for (tabular, graph, seq), targets in pbar:
                    tabular = tabular.to(device)
                    # Graph data is already on device
                    seq = seq.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs, _, _, _ = model(tabular, graph, seq)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    if config['scheduler'] == 'OneCycleLR':
                        scheduler.step()
                    
                    train_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            all_probs = []
            
            with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Val)") as pbar:
                for (tabular, graph, seq), targets in pbar:
                    tabular = tabular.to(device)
                    # Graph data is already on device
                    seq = seq.to(device)
                    targets = targets.to(device)
                    
                    outputs, _, _, _ = model(tabular, graph, seq)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # For binary classification, save probability of positive class
                    if outputs.shape[1] == 2:
                        all_probs.extend(probs[:, 1].cpu().numpy())
                    
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate metrics
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, zero_division=0)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            
            # AUC is only valid for binary classification
            auc = 0.0
            if len(all_probs) > 0:
                try:
                    auc = roc_auc_score(all_targets, all_probs)
                except:
                    auc = 0.0
            
            # Save epoch metrics
            epoch_metric = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            epoch_metrics.append(epoch_metric)
            
            # Print metrics
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # Update LR scheduler (for schedulers that step based on validation metrics)
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(f1)
            elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
                scheduler.step()
            
            # Check if this is the best model so far
            if f1 > best_val_f1:
                best_val_f1 = f1
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            # Report to Optuna for pruning
            trial.report(f1, epoch)
            
            # Handle pruning based on intermediate results
            if trial.should_prune():
                pruned = True
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping
            if early_stop_count >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Add best metrics from this fold
        best_epoch_metric = max(epoch_metrics, key=lambda x: x['f1'])
        fold_metrics.append(best_epoch_metric)
    
    # Compute average metrics across folds
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'auc': np.mean([m['auc'] for m in fold_metrics])
    }
    
    print("\nAverage metrics across folds:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save trial results for later analysis
    trial_results = {
        'trial_number': trial.number,
        'params': config,
        'metrics': avg_metrics,
        'fold_metrics': fold_metrics
    }
    
    # Save to file
    with open(RESULTS_DIR / f"trial_{trial.number}_results.json", 'w') as f:
        json.dump(trial_results, f, indent=2, cls=NumpyEncoder)
    
    # Return the primary optimization metric
    return avg_metrics['f1']

# Helper class for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Objective function for Optuna
def objective(trial):
    """Objective function for hyperparameter optimization"""
    try:
        # Load data
        X_tabular_train, X_tabular_test, graph_data, seq_train, seq_test, y_train, y_test = load_processed_data()
        
        # Move graph data to device once
        if hasattr(graph_data, 'to'):
            graph_data = graph_data.to(device)
        
        # Return cross-validated performance
        return train_and_evaluate_with_cv(
            trial, X_tabular_train, graph_data, seq_train, y_train, 
            n_splits=3, low_resource=LOW_RESOURCE
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Return a very low score to indicate failure
        return 0.0

# Create study visualization functions
def create_study_visualizations(study, output_dir):
    """
    Create and save optimization visualizations
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save plots
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Optimization History
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(output_dir / "optimization_history.png"))
        fig.write_html(str(output_dir / "optimization_history.html"))
    except Exception as e:
        print(f"Error creating optimization history plot: {e}")
    
    # 2. Parameter Importance
    try:
        fig = plot_param_importances(study)
        fig.write_image(str(output_dir / "param_importances.png"))
        fig.write_html(str(output_dir / "param_importances.html"))
    except Exception as e:
        print(f"Error creating parameter importance plot: {e}")
    
    # 3. Parallel Coordinate
    try:
        fig = plot_parallel_coordinate(study)
        fig.write_image(str(output_dir / "parallel_coordinate.png"))
        fig.write_html(str(output_dir / "parallel_coordinate.html"))
    except Exception as e:
        print(f"Error creating parallel coordinate plot: {e}")
    
    # 4. Custom correlation heatmap (using matplotlib)
    try:
        trials_df = study.trials_dataframe()
        # Filter for parameters only
        params = [c for c in trials_df.columns if c.startswith('params_')]
        if params:
            plt.figure(figsize=(12, 10))
            sns.heatmap(trials_df[params].corr(), annot=True, cmap='coolwarm')
            plt.title('Parameter Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(str(output_dir / "param_correlation.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating parameter correlation plot: {e}")
    
    # 5. Value distributions for top parameters
    try:
        importance = optuna.importance.get_param_importances(study)
        top_params = list(importance.keys())[:5]  # Top 5 parameters
        
        if top_params:
            plt.figure(figsize=(15, 3 * len(top_params)))
            for i, param in enumerate(top_params):
                plt.subplot(len(top_params), 1, i+1)
                data = []
                values = []
                
                # Get values and scores for this parameter
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE and param in trial.params:
                        data.append(trial.value)
                        values.append(trial.params[param])
                
                if isinstance(values[0], (int, float)):
                    # For numeric parameters
                    plt.scatter(values, data, alpha=0.7)
                    plt.xlabel(param)
                    plt.ylabel('F1 Score')
                    
                    # Add trend line
                    if len(set(values)) > 2:  # Only if we have more than 2 unique values
                        z = np.polyfit(values, data, 1)
                        p = np.poly1d(z)
                        plt.plot(sorted(values), p(sorted(values)), "r--", alpha=0.8)
                else:
                    # For categorical parameters
                    df = pd.DataFrame({'value': values, 'score': data})
                    sns.boxplot(x='value', y='score', data=df)
                    plt.xlabel(param)
                    plt.ylabel('F1 Score')
            
            plt.tight_layout()
            plt.savefig(str(output_dir / "param_distributions.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating parameter distribution plots: {e}")

# Function to generate publication-ready tables and charts
def generate_publication_materials(study, output_dir):
    """
    Generate tables and charts suitable for academic publications
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save results
    """
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Best parameters table (LaTeX format)
    try:
        best_params = study.best_params
        with open(output_dir / "best_params_table.tex", "w") as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Optimal Hyperparameters for MAGNET Model}\n")
            f.write("\\label{tab:best_params}\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\toprule\n")
            f.write("Hyperparameter & Optimal Value \\\\\n")
            f.write("\\midrule\n")
            
            for param, value in best_params.items():
                # Format parameter name for LaTeX
                param_name = param.replace("_", "\\_")
                f.write(f"{param_name} & {value} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    except Exception as e:
        print(f"Error generating best parameters table: {e}")
    
    # 2. Parameter importance table (LaTeX format)
    try:
        importance = optuna.importance.get_param_importances(study)
        with open(output_dir / "param_importance_table.tex", "w") as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Hyperparameter Importance for MAGNET Model}\n")
            f.write("\\label{tab:param_importance}\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\toprule\n")
            f.write("Hyperparameter & Importance Score \\\\\n")
            f.write("\\midrule\n")
            
            for param, score in importance.items():
                # Format parameter name for LaTeX
                param_name = param.replace("_", "\\_")
                f.write(f"{param_name} & {score:.4f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    except Exception as e:
        print(f"Error generating parameter importance table: {e}")
    
    # 3. Performance metrics for best model (LaTeX format)
    try:
        # Load results from the best trial
        best_trial = study.best_trial
        best_trial_results_path = RESULTS_DIR / f"trial_{best_trial.number}_results.json"
        
        if best_trial_results_path.exists():
            with open(best_trial_results_path, "r") as f:
                best_results = json.load(f)
            
            metrics = best_results.get("metrics", {})
            
            with open(output_dir / "performance_metrics_table.tex", "w") as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Performance Metrics for Optimized MAGNET Model}\n")
                f.write("\\label{tab:performance_metrics}\n")
                f.write("\\begin{tabular}{lc}\n")
                f.write("\\toprule\n")
                f.write("Metric & Value \\\\\n")
                f.write("\\midrule\n")
                
                for metric, value in metrics.items():
                    # Format metric name for LaTeX
                    metric_name = metric.capitalize().replace("_", "\\_")
                    f.write(f"{metric_name} & {value:.4f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
    except Exception as e:
        print(f"Error generating performance metrics table: {e}")
    
    # 4. CSV exports for all trials (for further analysis)
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_dir / "all_trials.csv", index=False)
    except Exception as e:
        print(f"Error exporting trials data to CSV: {e}")

def main():
    """Main execution function for hyperparameter optimization"""
    start_time = time.time()
    
    # Set up timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"study_{timestamp}"
    results_path.mkdir(exist_ok=True)
    
    print(f"{'='*80}")
    print(f"MAGNET Model Hyperparameter Optimization")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {results_path}")
    print(f"{'='*80}")
    
    # Configure the number of trials based on available resources
    if LOW_RESOURCE:
        n_trials = 20  # Fewer trials for limited resources
        print("Running in low-resource mode with 20 trials")
    else:
        n_trials = 50  # More trials for better results
        print("Running in standard mode with 50 trials")
    
    # Configure pruning to save resources
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    # Create and run the study
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=f"magnet_optimization_{timestamp}"
    )
    
    try:
        study.optimize(objective, n_trials=n_trials, timeout=None)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        # Even if interrupted, generate visualizations and reports
        duration = time.time() - start_time
        print(f"\nOptimization completed in {duration/60:.2f} minutes")
        
        # Save study
        with open(results_path / "study.pkl", "wb") as f:
            pickle.dump(study, f)
        
        # Print and save best results
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Trial number: {trial.number}")
        print(f"  F1 score: {trial.value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        
        # Save best configuration
        with open(results_path / "best_config.json", "w") as f:
            json.dump(trial.params, f, indent=2)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        create_study_visualizations(study, results_path / "visualizations")
        
        # Generate publication materials
        print("Generating publication materials...")
        generate_publication_materials(study, results_path / "publication")
        
        print(f"\nAll results saved to {results_path}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Check for import errors before starting
    try:
        import pickle  # Added here to ensure it's available for study saving
        main()
    except ImportError as e:
        print(f"Error: Missing required package: {str(e)}")
        print("Please install all requirements with: pip install -r requirements.txt")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc() 