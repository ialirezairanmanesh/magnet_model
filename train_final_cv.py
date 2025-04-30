#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: آموزش با Cross Validation
این اسکریپت مدل MAGNET را با 5-fold cross validation آموزش می‌دهد
و نتایج را به صورت خلاصه گزارش می‌دهد.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from magnet_model import MAGNET, MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"استفاده از دستگاه: {device}")

# تنظیم seed برای تکرارپذیری
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def train_epoch(model, train_loader, criterion, optimizer, device, ssl_weight=0.1):
    """آموزش یک اپوک"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc="آموزش", leave=False) as pbar:
        for batch in pbar:
            (tabular, graph, seq), targets = batch
            tabular = tabular.to(device)
            if hasattr(graph, 'to'):
                graph = graph.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # خروجی مدل
            logits, ssl_output, _, _ = model(tabular, graph, seq)
            
            # محاسبه خطا
            main_loss = criterion(logits, targets)
            
            # محاسبه خطای خودنظارتی
            if ssl_weight > 0:
                combined = torch.cat([tabular, seq], dim=1)
                if ssl_output.shape[1] != combined.shape[1]:
                    if not hasattr(model, 'ssl_adapter'):
                        model.ssl_adapter = nn.Linear(ssl_output.shape[1], combined.shape[1]).to(device)
                        print(f"\nتطبیق ابعاد SSL: {ssl_output.shape} -> {combined.shape}")
                    adapted_output = model.ssl_adapter(ssl_output)
                    ssl_loss = nn.MSELoss()(adapted_output, combined)
                else:
                    ssl_loss = nn.MSELoss()(ssl_output, combined)
                loss = main_loss + ssl_weight * ssl_loss
            else:
                loss = main_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * correct / total:.2f}%"
            )
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion, device):
    """ارزیابی مدل"""
    model.eval()
    total_loss = 0
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad(), tqdm(val_loader, desc="ارزیابی", leave=False) as pbar:
        for batch in pbar:
            (tabular, graph, seq), targets = batch
            tabular = tabular.to(device)
            if hasattr(graph, 'to'):
                graph = graph.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            logits, _, _, _ = model(tabular, graph, seq)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs)
    }
    
    return metrics

def train_fold(fold, n_splits, train_idx, val_idx, dataset, config, device):
    """آموزش برای یک fold"""
    # ایجاد دیتاست‌های train و validation
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # ایجاد مدل
    model = MAGNET(
        tabular_dim=config['tabular_dim'],
        graph_node_dim=config['graph_node_dim'],
        graph_edge_dim=config['graph_edge_dim'],
        seq_vocab_size=config['seq_vocab_size'],
        seq_max_len=config['seq_max_len'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    ).to(device)
    
    # تنظیم بهینه‌ساز و scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # آموزش
    best_val_f1 = 0
    best_metrics = None
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nFold {fold+1}/{n_splits}, Epoch {epoch+1}/{config['num_epochs']}")
        
        # آموزش
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            ssl_weight=config['alpha_ssl']
        )
        
        # ارزیابی
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # به‌روزرسانی scheduler
        scheduler.step()
        
        # نمایش نتایج
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # ذخیره بهترین مدل
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = val_metrics
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # ذخیره بهترین مدل برای این fold
    if best_model_state is not None:
        os.makedirs(f"results/cv_results_{timestamp}/models", exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'metrics': best_metrics
        }, f"results/cv_results_{timestamp}/models/fold_{fold}_best_model.pth")
    
    return best_metrics, best_model_state

def main():
    """تابع اصلی"""
    start_time = time.time()
    global timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # بارگذاری داده‌ها
    print("بارگذاری داده‌ها...")
    X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test = load_processed_data()
    
    # تبدیل به numpy array
    y_train = np.array(y_train)
    
    # تنظیمات مدل (بهترین پارامترها از Pirates و Optuna)
    config = {
        # پارامترهای معماری
        'embedding_dim': 32,
        'num_heads': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.2,
        
        # پارامترهای آموزش
        'batch_size': 16,
        'learning_rate': 0.0018984960342824176,
        'weight_decay': 0.0010956175512645931,
        'num_epochs': 20,  # افزایش تعداد اپوک‌ها
        'alpha_ssl': 0.1,
        
        # پارامترهای داده
        'tabular_dim': X_tabular_train.shape[1],
        'graph_node_dim': graph_data.x.shape[1],
        'graph_edge_dim': graph_data.edge_attr.shape[1] if hasattr(graph_data, 'edge_attr') else 0,
        'seq_vocab_size': 1000,
        'seq_max_len': 100,
        'num_classes': 2
    }
    
    # ایجاد دیتاست
    dataset = MultiModalDataset(X_tabular_train, graph_data, seq_data_train, y_train)
    
    # تنظیمات cross validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # نتایج هر fold
    fold_results = []
    fold_models = []
    
    # اجرای cross validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tabular_train, y_train)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*50}")
        
        # آموزش و ارزیابی fold
        fold_metrics, fold_model = train_fold(fold, n_splits, train_idx, val_idx, dataset, config, device)
        fold_results.append(fold_metrics)
        fold_models.append(fold_model)
        
        # نمایش نتایج fold
        print(f"\nنتایج Fold {fold+1}:")
        for metric, value in fold_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # محاسبه میانگین و انحراف معیار نتایج
    final_results = {}
    for metric in fold_results[0].keys():
        values = [result[metric] for result in fold_results]
        final_results[f'mean_{metric}'] = np.mean(values)
        final_results[f'std_{metric}'] = np.std(values)
    
    # انتخاب بهترین مدل بر اساس F1 score
    best_fold_idx = np.argmax([result['f1'] for result in fold_results])
    best_model_state = fold_models[best_fold_idx]
    
    # ذخیره بهترین مدل نهایی
    torch.save({
        'model_state_dict': best_model_state,
        'metrics': fold_results[best_fold_idx],
        'config': config
    }, f"results/cv_results_{timestamp}/magnet_best_model.pth")
    
    # نمایش نتایج نهایی
    print("\n" + "="*50)
    print("نتایج نهایی Cross Validation")
    print("="*50)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric}: {final_results[f'mean_{metric}']:.4f} ± {final_results[f'std_{metric}']:.4f}")
    
    # ذخیره نتایج
    results_dir = f"results/cv_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # ذخیره تنظیمات و نتایج
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(os.path.join(results_dir, "results.json"), 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'final_results': final_results,
            'best_fold': best_fold_idx
        }, f, indent=4)
    
    print(f"\nنتایج در {results_dir} ذخیره شدند")
    print(f"زمان کل اجرا: {time.time() - start_time:.2f} ثانیه")

if __name__ == "__main__":
    main() 