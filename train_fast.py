"""
آموزش سریع مدل MAGNET برای مرحله توسعه
این اسکریپت با کاهش پیچیدگی و اندازه مدل، امکان تست سریع را فراهم می‌کند
"""

import torch
import numpy as np
import time
from datetime import timedelta
import argparse
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# وارد کردن مدل سبک و داده‌ها
from fast_magnet import FastMAGNET
from magnet_model import MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تشخیص دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='آموزش سریع مدل MAGNET برای توسعه')
    parser.add_argument('--percentage', type=int, default=1, 
                        help='درصد داده‌ها برای آموزش (پیشنهاد: 1 تا 2)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='تعداد اپوک‌ها برای آموزش')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f" آموزش سریع مدل MAGNET روی {args.percentage}% از داده‌ها ")
    print(f"{'='*70}")
    
    # تنظیمات مدل سبک‌تر برای توسعه سریع
    config = {
        # پارامترهای مدل کوچک‌تر برای توسعه سریع
        'embedding_dim': 32,      # کاهش چشمگیر ابعاد embedding
        'num_heads': 2,           # کاهش تعداد سرها
        'num_layers': 2,          # کاهش تعداد لایه‌ها
        'dim_feedforward': 64,    # کاهش اندازه شبکه‌های feed-forward
        'dropout': 0.2,

        # پارامترهای آموزش سریع
        'batch_size': 128,           # بچ‌های بزرگتر برای کاهش زمان آموزش
        'num_epochs': args.epochs,
        'learning_rate': 0.001,      # نرخ یادگیری بالاتر
        'weight_decay': 0.01,
        'early_stop_patience': 2,    # توقف سریع‌تر

        # پارامترهای داده
        'seq_max_len': 100,
        'seq_vocab_size': 1000,
        'num_classes': 2,
    }
    
    os.makedirs('models', exist_ok=True)
    
    print(f"\nپارامترهای آموزش سریع:")
    print(f"  درصد داده‌ها: {args.percentage}% (بسیار کم برای توسعه سریع)")
    print(f"  تعداد اپوک‌ها: {config['num_epochs']}")
    print(f"  اندازه مدل: کوچک (حدود 200K پارامتر به جای 3.3M)")
    
    # زمان‌سنجی
    start_time = time.time()
    
    # --- بارگذاری داده‌ها ---
    print("\nبارگذاری داده‌ها...")
    X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test = load_processed_data()
    
    # استفاده از درصد کوچکی از داده‌ها
    print(f"نمونه‌های اولیه: {len(y_train)} آموزش, {len(y_test)} تست")
    
    sample_size_train = max(int(len(y_train) * args.percentage / 100), 50)  # حداقل 50 نمونه
    sample_size_test = max(int(len(y_test) * args.percentage / 100), 20)    # حداقل 20 نمونه
    
    indices_train = np.random.choice(len(y_train), sample_size_train, replace=False)
    indices_test = np.random.choice(len(y_test), sample_size_test, replace=False)
    
    X_tabular_train_sampled = X_tabular_train[indices_train]
    seq_data_train_sampled = seq_data_train[indices_train]
    y_train_sampled = y_train[indices_train]
    
    X_tabular_test_sampled = X_tabular_test[indices_test]
    seq_data_test_sampled = seq_data_test[indices_test]
    y_test_sampled = y_test[indices_test]
    
    print(f"نمونه‌های نهایی: {len(y_train_sampled)} آموزش, {len(y_test_sampled)} تست")
    
    # --- نرمال‌سازی داده‌ها ---
    scaler = StandardScaler()
    X_tabular_train_scaled = scaler.fit_transform(X_tabular_train_sampled)
    X_tabular_test_scaled = scaler.transform(X_tabular_test_sampled)
    
    # --- ایجاد دیتاست‌ها ---
    train_dataset = MultiModalDataset(X_tabular_train_scaled, graph_data, seq_data_train_sampled, y_train_sampled)
    test_dataset = MultiModalDataset(X_tabular_test_scaled, graph_data, seq_data_test_sampled, y_test_sampled)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    
    # --- آماده‌سازی مدل ---
    tabular_dim = X_tabular_train.shape[1]
    graph_node_dim = graph_data.x.shape[1]
    graph_edge_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0
    seq_vocab_size = config['seq_vocab_size']
    
    model = FastMAGNET(
        tabular_dim=tabular_dim,
        graph_node_dim=graph_node_dim,
        graph_edge_dim=graph_edge_dim,
        seq_vocab_size=seq_vocab_size,
        seq_max_len=config['seq_max_len'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    ).to(device)
    
    # شمارش تعداد پارامترها
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"مدل با {total_params:,} پارامتر آماده شد (حدود {total_params/1000000:.2f}M)")
    
    # --- آموزش مدل ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    print(f"\n--- شروع آموزش سریع برای {config['num_epochs']} اپوک ---")
    
    best_val_f1 = -1
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        # آموزش
        model.train()
        total_train_loss = 0
        
        for i, ((tabular, graph, seq), targets) in enumerate(train_loader):
            tabular = tabular.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits, _, _, _ = model(tabular, graph, seq)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # گزارش پیشرفت هر 5 بچ
            if (i+1) % 5 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end="\r")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ارزیابی
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for (tabular, graph, seq), targets in test_loader:
                tabular = tabular.to(device)
                seq = seq.to(device)
                
                logits, _, _, _ = model(tabular, graph, seq)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # محاسبه معیارها
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"*** مدل جدید با F1={val_f1:.4f} ذخیره شد ***")
    
    # --- ارزیابی نهایی ---
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for (tabular, graph, seq), targets in test_loader:
            tabular = tabular.to(device)
            seq = seq.to(device)
            
            logits, _, _, _ = model(tabular, graph, seq)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # محاسبه و نمایش نتایج
    final_accuracy = accuracy_score(all_targets, all_preds)
    final_precision = precision_score(all_targets, all_preds, zero_division=0)
    final_recall = recall_score(all_targets, all_preds, zero_division=0)
    final_f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    results = {
        'Accuracy': final_accuracy,
        'Precision': final_precision,
        'Recall': final_recall,
        'F1 Score': final_f1
    }
    
    # زمان اجرا
    elapsed_time = time.time() - start_time
    
    print(f"\nزمان اجرا: {timedelta(seconds=elapsed_time)}")
    print("\nماتریس اغتشاش:")
    print(cm)
    print("\nنتایج نهایی:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # ذخیره مدل
    model_path = f'models/fast_magnet_{args.percentage}percent.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results
    }, model_path)
    
    print(f"\nمدل با موفقیت در {model_path} ذخیره شد.") 