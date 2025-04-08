"""
آموزش مدل متوسط MAGNET: تعادل بین سرعت و دقت برای مرحله توسعه
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
from sklearn.utils.class_weight import compute_class_weight

# وارد کردن مدل متوسط و داده‌ها
from medium_magnet import MediumMAGNET
from magnet_model import MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تشخیص دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='آموزش مدل متوسط MAGNET')
    parser.add_argument('--percentage', type=int, default=5, 
                        help='درصد داده‌ها برای آموزش (پیشنهاد: 5 تا 10)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='تعداد اپوک‌ها برای آموزش')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f" آموزش مدل متوسط MAGNET روی {args.percentage}% از داده‌ها ")
    print(f"{'='*70}")
    
    # تنظیمات مدل متوسط
    config = {
        # پارامترهای مدل با پیچیدگی متوسط
        'embedding_dim': 64,  # بیشتر از مدل سریع، کمتر از مدل کامل
        'dropout': 0.3,
        
        # پارامترهای آموزش
        'batch_size': 64,
        'num_epochs': args.epochs,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'early_stop_patience': 5,
        'alpha_ssl': 0.1,  # فعال کردن یادگیری خودنظارت با وزن کم
        
        # پارامترهای داده
        'seq_max_len': 100,
        'seq_vocab_size': 1000,
        'num_classes': 2,
    }
    
    os.makedirs('models', exist_ok=True)
    
    print(f"\nپارامترهای آموزش مدل متوسط:")
    print(f"  درصد داده‌ها: {args.percentage}%")
    print(f"  تعداد اپوک‌ها: {config['num_epochs']}")
    print(f"  بعد embedding: {config['embedding_dim']}")
    print(f"  نرخ یادگیری: {config['learning_rate']}")
    
    # زمان‌سنجی
    start_time = time.time()
    
    # --- بارگذاری داده‌ها ---
    print("\nبارگذاری داده‌ها...")
    X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test = load_processed_data()
    
    # استفاده از درصد مناسب از داده‌ها
    print(f"نمونه‌های اولیه: {len(y_train)} آموزش, {len(y_test)} تست")
    
    # نمونه‌گیری طبقه‌بندی شده برای حفظ توزیع کلاس‌ها
    indices_class0 = np.where(y_train == 0)[0]
    indices_class1 = np.where(y_train == 1)[0]
    
    # تعیین تعداد نمونه از هر کلاس (با حفظ نسبت)
    total_samples = max(int(len(y_train) * args.percentage / 100), 100)
    class0_ratio = len(indices_class0) / len(y_train)
    samples_class0 = max(int(total_samples * class0_ratio), 10)  # حداقل 10 نمونه
    samples_class1 = total_samples - samples_class0
    
    # نمونه‌گیری از هر کلاس
    sampled_indices_class0 = np.random.choice(indices_class0, min(samples_class0, len(indices_class0)), replace=False)
    sampled_indices_class1 = np.random.choice(indices_class1, min(samples_class1, len(indices_class1)), replace=False)
    
    # ترکیب نمونه‌ها
    indices_train = np.concatenate([sampled_indices_class0, sampled_indices_class1])
    np.random.shuffle(indices_train)  # برای اطمینان از ترتیب تصادفی
    
    X_tabular_train_sampled = X_tabular_train[indices_train]
    seq_data_train_sampled = seq_data_train[indices_train]
    y_train_sampled = y_train[indices_train]
    
    X_tabular_test_sampled = X_tabular_test[indices_test]
    seq_data_test_sampled = seq_data_test[indices_test]
    y_test_sampled = y_test[indices_test]
    
    print(f"نمونه‌های نهایی: {len(y_train_sampled)} آموزش, {len(y_test_sampled)} تست")
    
    # توزیع کلاس‌ها
    unique_train, counts_train = np.unique(y_train_sampled, return_counts=True)
    train_class_dist = dict(zip(unique_train, counts_train))
    print(f"توزیع کلاس‌ها (آموزش): {train_class_dist}")
    
    # --- محاسبه وزن کلاس‌ها ---
    class_weights = compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=y_train_sampled
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"وزن کلاس‌ها: {class_weights.cpu().numpy()}")
    
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
    
    model = MediumMAGNET(
        tabular_dim=tabular_dim,
        graph_node_dim=graph_node_dim,
        graph_edge_dim=graph_edge_dim,
        seq_vocab_size=seq_vocab_size,
        seq_max_len=config['seq_max_len'],
        embedding_dim=config['embedding_dim'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    ).to(device)
    
    # شمارش تعداد پارامترها
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"مدل با {total_params:,} پارامتر آماده شد (حدود {total_params/1000000:.2f}M)")
    
    # --- آموزش مدل ---
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    self_supervised_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    print(f"\n--- شروع آموزش مدل متوسط برای {config['num_epochs']} اپوک ---")
    
    best_val_f1 = -1
    best_model_state = None
    early_stop_counter = 0
    alpha = config['alpha_ssl']
    
    for epoch in range(config['num_epochs']):
        # آموزش
        model.train()
        total_train_loss = 0
        total_ssl_loss = 0
        
        for i, ((tabular, graph, seq), targets) in enumerate(train_loader):
            tabular = tabular.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            logits, ssl_output, _, _ = model(tabular, graph, seq)
            
            # محاسبه لاس طبقه‌بندی
            classification_loss = criterion(logits, targets)
            
            # محاسبه لاس خودنظارت
            original_features = torch.cat([tabular, 
                                          graph_data.x.mean(dim=0).expand(tabular.size(0), -1), 
                                          seq.float().mean(dim=1)], dim=1)
            self_supervised_loss = self_supervised_criterion(ssl_output, original_features)
            
            # ترکیب لاس‌ها
            loss = classification_loss + alpha * self_supervised_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += classification_loss.item()
            total_ssl_loss += self_supervised_loss.item()
            
            # گزارش پیشرفت
            if (i+1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end="\r")
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_ssl_loss = total_ssl_loss / len(train_loader)
        
        # ارزیابی
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for (tabular, graph, seq), targets in test_loader:
                tabular = tabular.to(device)
                seq = seq.to(device)
                
                logits, _, _, _ = model(tabular, graph, seq)
                loss = criterion(logits, targets.to(device))
                total_val_loss += loss.item()
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # محاسبه معیارها
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        avg_val_loss = total_val_loss / len(test_loader)
        
        # چاپ نتایج
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"SSL Loss: {avg_ssl_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Val F1: {val_f1:.4f}")
        
        # به‌روزرسانی scheduler
        scheduler.step(val_f1)
        
        # ذخیره بهترین مدل
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            print(f"*** مدل جدید با F1={val_f1:.4f} ذخیره شد ***")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['early_stop_patience']:
                print(f"توقف زودهنگام در اپوک {epoch+1} - {early_stop_counter} اپوک بدون بهبود")
                break
    
    # --- ارزیابی نهایی ---
    if best_model_state:
        print("\nبارگذاری بهترین مدل برای ارزیابی نهایی...")
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
    model_path = f'models/medium_magnet_{args.percentage}percent_f1_{final_f1:.4f}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': config,
        'results': results
    }, model_path)
    
    print(f"\nمدل با موفقیت در {model_path} ذخیره شد.") 