#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: تست نهایی مدل روی داده‌های تست
این اسکریپت بهترین مدل آموزش دیده را روی داده‌های تست ارزیابی می‌کند
و نتایج را به صورت کامل گزارش می‌دهد.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from magnet_model import MAGNET, MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"استفاده از دستگاه: {device}")

def evaluate_test(model, test_loader, criterion, device):
    """ارزیابی مدل روی داده‌های تست"""
    model.eval()
    total_loss = 0
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    # محاسبه معیارها
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs)
    }
    
    # محاسبه ماتریس درهم‌ریختگی
    cm = confusion_matrix(all_targets, all_preds)
    
    # گزارش طبقه‌بندی
    report = classification_report(all_targets, all_preds, output_dict=True)
    
    return metrics, cm, report, all_targets, all_preds, all_probs

def plot_confusion_matrix(cm, save_path=None):
    """رسم ماتریس درهم‌ریختگی"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودار ماتریس درهم‌ریختگی در {save_path} ذخیره شد")
    
    plt.show()

def main():
    """تابع اصلی"""
    start_time = time.time()
    
    # بارگذاری داده‌ها
    print("بارگذاری داده‌ها...")
    X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test = load_processed_data()
    
    # تبدیل به numpy array
    y_test = np.array(y_test)
    
    # تنظیمات مدل (بهترین پارامترها از Pirates و Optuna)
    config = {
        # پارامترهای معماری
        'embedding_dim': 32,
        'num_heads': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.2,
        
        # پارامترهای داده
        'tabular_dim': X_tabular_test.shape[1],
        'graph_node_dim': graph_data.x.shape[1],
        'graph_edge_dim': graph_data.edge_attr.shape[1] if hasattr(graph_data, 'edge_attr') else 0,
        'seq_vocab_size': 1000,
        'seq_max_len': 100,
        'num_classes': 2
    }
    
    # ایجاد دیتاست تست
    test_dataset = MultiModalDataset(X_tabular_test, graph_data, seq_data_test, y_test)
    
    # ایجاد دیتالودر تست
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
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
    
    # بارگذاری بهترین مدل
    model_path = "results/magnet_final_20250425_133233/magnet_best_model.pth"  # مسیر مدل بهترین
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # حذف وزن‌های اضافی SSL adapter
    state_dict = checkpoint['model_state_dict']
    keys_to_remove = [k for k in state_dict.keys() if k.startswith('ssl_adapter')]
    for k in keys_to_remove:
        del state_dict[k]
    
    model.load_state_dict(state_dict)
    print(f"مدل از {model_path} بارگذاری شد")
    
    # ارزیابی روی داده‌های تست
    criterion = nn.CrossEntropyLoss()
    metrics, cm, report, targets, preds, probs = evaluate_test(model, test_loader, criterion, device)
    
    # نمایش نتایج
    print("\n" + "="*50)
    print("نتایج ارزیابی روی داده‌های تست")
    print("="*50)
    
    print("\nمعیارهای اصلی:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nگزارش طبقه‌بندی:")
    print(classification_report(targets, preds))
    
    # ذخیره نتایج
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/test_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # ذخیره نتایج در فایل JSON
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    with open(os.path.join(results_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # رسم و ذخیره ماتریس درهم‌ریختگی
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, save_path=cm_path)
    
    print(f"\nنتایج در {results_dir} ذخیره شدند")
    print(f"زمان کل اجرا: {time.time() - start_time:.2f} ثانیه")

if __name__ == "__main__":
    main() 