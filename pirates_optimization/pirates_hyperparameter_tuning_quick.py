#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: Hyperparameter Optimization with Pirates Algorithm (Quick Version)
=========================================================
This is a quick version of the optimization script with limited trials.
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

# Set paths
BASE_DIR = Path('/home/alireza/Documents/final_magnet/magnet_model')
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results/pirates_optimization'
MODELS_DIR = BASE_DIR / 'models'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# Check memory capacity
if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
    LOW_RESOURCE = gpu_mem_gb < 4
else:
    LOW_RESOURCE = True
    print("Running on CPU. Using low-resource mode.")

# Custom collate function
def custom_collate(batch):
    tabular_data = torch.stack([item[0][0] for item in batch])
    graph_data = batch[0][0][1]
    seq_data = torch.stack([item[0][2] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return (tabular_data, graph_data, seq_data), targets

# Dataset class
class MultiModalDataset(Dataset):
    def __init__(self, X_tabular, graph_data, seq_data, y):
        self.X_tabular = X_tabular
        self.graph_data = graph_data
        self.seq_data = seq_data
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

# Configuration mapping
def get_config_from_position(position):
    embedding_dims = [16, 32, 64, 128]
    num_heads_options = [2, 4, 8, 16]
    dim_feedforward_options = [64, 128, 256, 512]
    batch_sizes = [16, 32, 64, 128]
    
    config = {
        'embedding_dim': embedding_dims[min(int(position[0] * len(embedding_dims)), len(embedding_dims)-1)],
        'num_heads': num_heads_options[min(int(position[1] * len(num_heads_options)), len(num_heads_options)-1)],
        'num_layers': min(int(position[2] * 4) + 1, 4),
        'dim_feedforward': dim_feedforward_options[min(int(position[3] * len(dim_feedforward_options)), len(dim_feedforward_options)-1)],
        'dropout': position[4] * 0.5,
        'batch_size': batch_sizes[min(int(position[5] * len(batch_sizes)), len(batch_sizes)-1)],
        'learning_rate': 10 ** (position[6] * (np.log10(1e-2) - np.log10(1e-4)) + np.log10(1e-4)),
        'weight_decay': 10 ** (position[7] * (np.log10(1e-1) - np.log10(1e-5)) + np.log10(1e-5)),
        'num_epochs': 1 if '--test' in sys.argv else (3 if LOW_RESOURCE else 5)
    }
    
    valid_heads = [h for h in num_heads_options if h <= config['embedding_dim'] and config['embedding_dim'] % h == 0]
    if valid_heads:
        config['num_heads'] = valid_heads[min(int(position[1] * len(valid_heads)), len(valid_heads)-1)]
    else:
        config['num_heads'] = 2
    
    return config

# Training and evaluation function
def train_and_evaluate_magnet(position, X_tabular_train, X_tabular_test, graph_data, seq_train, seq_test, y_train, y_test):
    config = get_config_from_position(position)
    
    print(f"\nTesting configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    data_percentage = 2 if '--test' in sys.argv else 50  # Changed to 50% for normal mode
    
    if data_percentage < 100:
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
    
    train_dataset = MultiModalDataset(X_tabular_train_subset, graph_data, seq_train_subset, y_train_subset)
    test_dataset = MultiModalDataset(X_tabular_test_subset, graph_data, seq_test_subset, y_test_subset)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate)
    
    model = MAGNET(
        tabular_dim=X_tabular_train_subset.size(1),
        graph_node_dim=graph_data.x.size(1),
        graph_edge_dim=graph_data.edge_attr.size(1) if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else 0,
        seq_vocab_size=1000,
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        seq_max_len=seq_train_subset.size(1)
    ).to(device)
    
    y_train_tensor = train_dataset.y
    class_counts = torch.bincount(y_train_tensor)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=config['learning_rate'] * 0.01
    )
    
    best_val_f1 = 0.0
    all_epoch_metrics = []
    
    print("\nTraining progress:")
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        
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
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
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
        
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
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
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={epoch_metrics['val_loss']:.4f}")
        print(f"Metrics: F1={val_f1:.4f}, Accuracy={val_accuracy:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}")
        
        scheduler.step()
    
    best_epoch = max(all_epoch_metrics, key=lambda x: x['f1'])
    
    metrics = {
        'f1': best_epoch['f1'],
        'accuracy': best_epoch['accuracy'],
        'precision': best_epoch['precision'],
        'recall': best_epoch['recall'],
        'config': config,
        'epochs_data': all_epoch_metrics
    }
    
    return -best_epoch['f1'], metrics

# JSON encoder
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
            return str(type(obj))
        return super(NumpyEncoder, self).default(obj)

# Optimization class
class HyperparameterOptimization:
    def __init__(self, dimensions=8, resume=True):
        self.dimensions = dimensions
        self.best_metrics = None
        self.best_error = float('inf')
        self.data_loaded = False
        self.all_trials = []
        self.no_improvement_count = 0
        self.search_radius = 0.3
        self.min_search_radius = 0.05
        self.radius_decay = 0.95
        
        # Clear stopping conditions
        self.target_f1 = 0.99  # Target F1 score to achieve
        self.max_no_improvement = 5  # Maximum number of trials without improvement
        self.min_improvement = 0.001  # Minimum improvement threshold (0.1%)
        self.max_total_trials = 476  # Maximum total number of trials
        
        self.load_data()
        
        if resume:
            self.load_progress()
    
    def load_data(self):
        try:
            processed_data_dir = Path('/home/alireza/Documents/final_magnet/magnet_model/processed_data')
            
            print("Loading data from processed_data directory...")
            self.X_tabular_train = torch.load(processed_data_dir / 'X_tabular_train.pt', weights_only=False)
            self.X_tabular_test = torch.load(processed_data_dir / 'X_tabular_test.pt', weights_only=False)
            self.graph_data = torch.load(processed_data_dir / 'graph_data_processed.pt', weights_only=False)
            self.seq_train = torch.load(processed_data_dir / 'seq_train.pt', weights_only=False)
            self.seq_test = torch.load(processed_data_dir / 'seq_test.pt', weights_only=False)
            self.y_train = torch.load(processed_data_dir / 'y_train.pt', weights_only=False)
            self.y_test = torch.load(processed_data_dir / 'y_test.pt', weights_only=False)
            
            self.graph_data = self.graph_data.to(device)
            
            print(f"Data loaded successfully: {self.X_tabular_train.shape[0]} training samples, {self.X_tabular_test.shape[0]} test samples")
            self.data_loaded = True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def load_progress(self):
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
        config = get_config_from_position(position)
        
        # First evaluate the current position
        current_neg_f1, current_metrics = train_and_evaluate_magnet(
            position, self.X_tabular_train, self.X_tabular_test,
            self.graph_data, self.seq_train, self.seq_test,
            self.y_train, self.y_test
        )
        
        best_position = position
        best_neg_f1 = current_neg_f1
        best_metrics = current_metrics
        
        # Enhanced local search for better exploration
        if len(self.all_trials) > 0 and self.best_error < float('inf'):
            if np.random.random() < 0.5:
                for _ in range(3):  # Try 3 local search attempts
                    alt_position = self.local_search(best_position)
                    alt_neg_f1, alt_metrics = train_and_evaluate_magnet(
                        alt_position, self.X_tabular_train, self.X_tabular_test,
                        self.graph_data, self.seq_train, self.seq_test,
                        self.y_train, self.y_test
                    )
                    if alt_neg_f1 < best_neg_f1:
                        best_position = alt_position
                        best_neg_f1 = alt_neg_f1
                        best_metrics = alt_metrics
        
        # Update config with best found position
        config = get_config_from_position(best_position)
        config['X_tabular_train'] = self.X_tabular_train
        config['X_tabular_test'] = self.X_tabular_test
        config['graph_data'] = self.graph_data
        config['seq_train'] = self.seq_train
        config['seq_test'] = self.seq_test
        config['y_train'] = self.y_train
        config['y_test'] = self.y_test
        
        trial_number = len(self.all_trials) + 1
        
        # Check if maximum trials reached
        if trial_number > self.max_total_trials:
            print(f"\n⚠️ Maximum number of trials ({self.max_total_trials}) reached!")
            print("Stopping optimization...")
            raise Exception("Maximum trials reached")
        
        print(f"\n{'='*80}")
        print(f"Trial {trial_number}/{self.max_total_trials} - Starting evaluation")
        print(f"Search radius: {self.search_radius:.3f}")
        print(f"Current best F1: {-self.best_error:.4f}")
        print(f"Target F1: {self.target_f1:.4f}")
        print(f"Trials without improvement: {self.no_improvement_count}/{self.max_no_improvement}")
        print(f"{'='*80}")
        
        # Use the best found metrics from local search
        neg_f1 = best_neg_f1
        metrics = best_metrics
        
        trial_data = {
            'trial_number': trial_number,
            'position': best_position.tolist(),
            'neg_f1': neg_f1,
            'metrics': metrics
        }
        self.all_trials.append(trial_data)
        
        # Enhanced improvement checking with clear conditions
        if neg_f1 < self.best_error:
            improvement = (self.best_error - neg_f1) / self.best_error
            print(f"\n🌟 New best result! F1 score: {-neg_f1:.4f} (Improvement: {improvement*100:.2f}%)")
            self.best_error = neg_f1
            metrics['error'] = neg_f1
            metrics['position'] = best_position.tolist()
            metrics['config'] = config
            self.best_metrics = metrics
            self.no_improvement_count = 0
            
            # Check if target F1 score is achieved
            if -neg_f1 >= self.target_f1:
                print(f"\n🎯 Target F1 score ({self.target_f1}) achieved!")
                print("Stopping optimization...")
                raise Exception("Target F1 score achieved")
            
            # More aggressive radius increase on significant improvement
            if improvement > 0.01:
                self.search_radius = min(0.4, self.search_radius * 1.2)
        else:
            self.no_improvement_count += 1
            print(f"\nNo improvement for {self.no_improvement_count}/{self.max_no_improvement} trials")
            
            # Check if improvement is too small
            if self.no_improvement_count >= self.max_no_improvement:
                print(f"\n❌ No significant improvement for {self.max_no_improvement} consecutive trials")
                print("Stopping optimization...")
                raise Exception("Maximum trials without improvement reached")
            
            # Slower radius decrease for more exploration
            self.search_radius = max(self.min_search_radius, self.search_radius * self.radius_decay)
        
        self.save_progress()
        return neg_f1
    
    def local_search(self, position):
        """Enhanced local search with multiple perturbation scales"""
        local_position = position.copy()
        # Try different perturbation scales
        scales = [0.1, 0.05, 0.02]
        best_position = position.copy()
        best_f1 = float('inf')
        
        for scale in scales:
            temp_position = position.copy()
            for i in range(len(position)):
                perturbation = np.random.normal(0, self.search_radius * scale)
                temp_position[i] = np.clip(temp_position[i] + perturbation, 0, 1)
            
            # Quick evaluation
            temp_config = get_config_from_position(temp_position)
            temp_neg_f1, _ = train_and_evaluate_magnet(
                temp_position, self.X_tabular_train, self.X_tabular_test,
                self.graph_data, self.seq_train, self.seq_test,
                self.y_train, self.y_test
            )
            
            if temp_neg_f1 < best_f1:
                best_f1 = temp_neg_f1
                best_position = temp_position
        
        return best_position
    
    def save_progress(self):
        progress_file = RESULTS_DIR / f"optimization_progress.json"
        try:
            best_metrics_clean = None
            if self.best_metrics:
                best_metrics_clean = self.best_metrics.copy()
                if 'config' in best_metrics_clean:
                    config_clean = {k: v for k, v in best_metrics_clean['config'].items() 
                                 if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 
                                             'seq_train', 'seq_test', 'y_train', 'y_test']}
                    best_metrics_clean['config'] = config_clean
            
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

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"pirates_results_{timestamp}.json"
    
    parser = argparse.ArgumentParser(description='MAGNET Hyperparameter Optimization with Pirates (Quick Version)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous checkpoint')
    parser.add_argument('--no-resume', action='store_false', dest='resume', help='Start fresh optimization')
    parser.add_argument('--test', action='store_true', help='Run in test mode with minimal parameters')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"MAGNET Model Hyperparameter Optimization with Pirates Algorithm (Quick Version)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {results_file}")
    if args.resume:
        print(f"Attempting to resume from previous checkpoint")
    else:
        print(f"Starting fresh optimization (resume disabled)")
    print(f"{'='*80}")
    
    # Configure parameters for better results
    if args.test:
        num_ships = 2
        max_iter = 2
        print("Running in TEST mode with minimal parameters")
    else:
        num_ships = 5  # Increased from 2 to 5
        max_iter = 5   # Increased from 3 to 5
        print("Running in enhanced mode for better results")
    
    opt_func = HyperparameterOptimization(dimensions=8, resume=args.resume)
    
    completed_trials = len(opt_func.all_trials)
    if args.resume and completed_trials > 0:
        completed_iterations = completed_trials // num_ships
        remaining_iterations = max(1, max_iter - completed_iterations)
        print(f"Already completed {completed_trials} trials ({completed_iterations} iterations)")
        print(f"Remaining iterations: {remaining_iterations}")
        max_iter = remaining_iterations
    
    pirates = Pirates(
        func=opt_func,
        fmin=(0, 0, 0, 0, 0, 0, 0, 0),
        fmax=(1, 1, 1, 1, 1, 1, 1, 1),
        hr=0.2,
        ms=5,
        max_r=1,
        num_ships=num_ships,
        dimensions=8,
        max_iter=max_iter,
        max_wind=0.5,
        c={},
        top_ships=3,  # Increased for better exploration
        dynamic_sails=True,
        iteration_plots=False,
        quiet=False,
        sailing_radius=0.3,
        plundering_radius=0.1
    )
    
    print("\nStarting Pirates optimization...")
    print(f"Configuration: {num_ships} ships, {max_iter} iterations")
    
    try:
        best_position, best_cost, best_metrics = pirates.search()
        
        best_metrics = opt_func.best_metrics
        
        if best_metrics:
            best_f1 = -best_metrics['error'] if 'error' in best_metrics else 0
            best_config = best_metrics['config'] if 'config' in best_metrics else {}
            
            print(f"\n{'='*80}")
            print(f"Optimization completed in {(time.time() - start_time)/60:.2f} minutes")
            print(f"{'='*80}")
            print("\nBest hyperparameters:")
            for k, v in best_config.items():
                if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 'seq_train', 'seq_test', 'y_train', 'y_test']:
                    print(f"  {k}: {v}")
            print(f"Best F1 Score: {best_f1:.4f}")
            
            saveable_config = {k: str(v) if isinstance(v, torch.Tensor) else v for k, v in best_config.items() 
                             if k not in ['X_tabular_train', 'X_tabular_test', 'graph_data', 'seq_train', 'seq_test', 'y_train', 'y_test']}
            
            optimization_results = {
                'best_params': saveable_config,
                'best_f1_score': best_f1,
                'best_position': best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
                'timestamp': timestamp,
                'duration_minutes': (time.time() - start_time)/60,
                'all_trials': opt_func.all_trials
            }
            
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, cls=NumpyEncoder, indent=2)
            
            print(f"Results saved to {results_file}")
            
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No valid results found")
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
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