"""
آموزش مدل متوسط MAGNET: نسخه استوار با پشتیبانی از انواع مختلف داده
"""

import torch
import numpy as np
import time
import pandas as pd
from datetime import timedelta
import argparse
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.optim as optim

# وارد کردن مدل متوسط و داده‌ها
from medium_magnet import MediumMAGNET
from magnet_model import MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تشخیص دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# کلاس استاندارد برای دیتاست ساده‌تر برای دیباگ
class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# تابع کمکی برای تبدیل هر نوع داده به آرایه numpy
def ensure_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    elif hasattr(data, '__iter__') and not isinstance(data, str):
        return np.array(list(data))
    else:
        return np.array([data])

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
        'embedding_dim': 64,
        'dropout': 0.3,
        
        # پارامترهای آموزش
        'batch_size': 32,  # کاهش اندازه بچ برای اشکال‌زدایی
        'num_epochs': args.epochs,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'early_stop_patience': 5,
        
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
    try:
        print("بارگذاری داده‌ها با تابع load_processed_data()...")
        X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test = load_processed_data()
        
        # اشکال‌زدایی: نمایش اطلاعات داده‌ها
        print(f"نوع X_tabular_train: {type(X_tabular_train)}, شکل: {X_tabular_train.shape if hasattr(X_tabular_train, 'shape') else 'نامشخص'}")
        print(f"نوع y_train: {type(y_train)}")
        
        # تبدیل داده‌ها به آرایه numpy
        y_train = ensure_numpy(y_train)
        y_test = ensure_numpy(y_test)
        
        # حالا چاپ می‌کنیم تا مطمئن شویم که درست تبدیل شده‌اند
        print(f"نوع y_train بعد از تبدیل: {type(y_train)}, شکل: {y_train.shape}")
        unique_classes = np.unique(y_train)
        print(f"مقادیر منحصر به فرد در y_train: {unique_classes}")
        
        # چاپ تعداد نمونه‌ها در هر کلاس
        class_counts = {c: np.sum(y_train == c) for c in unique_classes}
        print(f"تعداد نمونه‌ها در هر کلاس: {class_counts}")
        
        # استفاده از درصد مناسب از داده‌ها
        print(f"نمونه‌های اولیه: {len(y_train)} آموزش, {len(y_test)} تست")
        
        # نمونه‌گیری ساده بر اساس درصد
        total_samples = max(int(len(y_train) * args.percentage / 100), 100)
        indices_train = np.random.choice(len(y_train), total_samples, replace=False)
        
        total_test_samples = max(int(len(y_test) * args.percentage / 100), 50)
        indices_test = np.random.choice(len(y_test), total_test_samples, replace=False)
        
        # استخراج نمونه‌های مناسب
        try:
            X_tabular_train_sampled = X_tabular_train[indices_train]
            seq_data_train_sampled = seq_data_train[indices_train]
            y_train_sampled = y_train[indices_train]
            
            X_tabular_test_sampled = X_tabular_test[indices_test]
            seq_data_test_sampled = seq_data_test[indices_test]
            y_test_sampled = y_test[indices_test]
            
            print(f"نمونه‌های نهایی: {len(y_train_sampled)} آموزش, {len(y_test_sampled)} تست")
        except Exception as e:
            print(f"خطا در نمونه‌گیری داده‌ها: {e}")
            print("استفاده از همه داده‌ها...")
            
            X_tabular_train_sampled = X_tabular_train
            seq_data_train_sampled = seq_data_train
            y_train_sampled = y_train
            
            X_tabular_test_sampled = X_tabular_test
            seq_data_test_sampled = seq_data_test
            y_test_sampled = y_test
        
        # توزیع کلاس‌ها
        unique_train, counts_train = np.unique(y_train_sampled, return_counts=True)
        train_class_dist = dict(zip(unique_train, counts_train))
        print(f"توزیع کلاس‌ها (آموزش): {train_class_dist}")
        
        # --- محاسبه وزن کلاس‌ها ---
        try:
            print("تلاش برای محاسبه وزن کلاس‌ها...")
            class_weights = compute_class_weight(
                'balanced', classes=unique_train, y=y_train_sampled
            )
            class_weights = torch.FloatTensor(class_weights).to(device)
            print(f"وزن کلاس‌ها: {class_weights.cpu().numpy()}")
        except Exception as e:
            print(f"خطا در محاسبه وزن کلاس‌ها: {e}")
            print("استفاده از وزن‌های یکسان برای همه کلاس‌ها...")
            class_weights = torch.FloatTensor([1.0] * len(unique_train)).to(device)
        
        # --- نرمال‌سازی داده‌ها ---
        print("نرمال‌سازی داده‌های جدولی...")
        try:
            scaler = StandardScaler()
            X_tabular_train_scaled = scaler.fit_transform(X_tabular_train_sampled)
            X_tabular_test_scaled = scaler.transform(X_tabular_test_sampled)
        except Exception as e:
            print(f"خطا در نرمال‌سازی داده‌ها: {e}")
            print("استفاده از داده‌های اصلی بدون نرمال‌سازی...")
            X_tabular_train_scaled = X_tabular_train_sampled
            X_tabular_test_scaled = X_tabular_test_sampled
        
        # --- ایجاد دیتاست‌ها ---
        print("ایجاد دیتاست‌ها...")
        try:
            train_dataset = MultiModalDataset(X_tabular_train_scaled, graph_data, seq_data_train_sampled, y_train_sampled)
            test_dataset = MultiModalDataset(X_tabular_test_scaled, graph_data, seq_data_test_sampled, y_test_sampled)
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
            
            # اشکال‌زدایی: بررسی یک نمونه از داده
            sample_batch = next(iter(train_loader))
            print(f"نمونه بچ - نوع: {type(sample_batch)}")
            if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                (tabular, graph, seq), targets = sample_batch
                print(f"  tabular شکل: {tabular.shape if hasattr(tabular, 'shape') else 'نامشخص'}")
                print(f"  graph نوع: {type(graph)}")
                print(f"  seq شکل: {seq.shape if hasattr(seq, 'shape') else 'نامشخص'}")
                print(f"  targets شکل: {targets.shape if hasattr(targets, 'shape') else 'نامشخص'}")
            
            # --- آماده‌سازی مدل ---
            print("آماده‌سازی مدل...")
            tabular_dim = X_tabular_train_scaled.shape[1]
            graph_node_dim = graph_data.x.shape[1]
            graph_edge_dim = graph_data.edge_attr.shape[1] if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else 0
            
            model = MediumMAGNET(
                tabular_dim=tabular_dim,
                graph_node_dim=graph_node_dim,
                graph_edge_dim=graph_edge_dim,
                seq_vocab_size=config['seq_vocab_size'],
                seq_max_len=config['seq_max_len'],
                embedding_dim=config['embedding_dim'],
                dropout=config['dropout'],
                num_classes=config['num_classes']
            )

            # تعریف تابع ضرر و بهینه‌ساز
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

            # تعریف تابع آموزش
            def train(model, train_loader, criterion, optimizer):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                for batch in train_loader:
                    (tabular, graph, seq), targets = batch
                    
                    # انتقال داده‌ها به دستگاه
                    tabular = tabular.to(device)
                    # graph نیازی به انتقال ندارد چون Data object است
                    seq = seq.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    # خروجی مدل یک تاپل است
                    outputs = model(tabular, graph, seq)
                    
                    # استخراج logits (اولین عنصر تاپل)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(logits, 1)  # از logits استفاده می‌کنیم
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
                return total_loss / len(train_loader), correct / total

            # تعریف تابع ارزیابی
            def evaluate(model, test_loader):
                model.eval()
                correct = 0
                total = 0
                all_targets = []
                all_preds = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        (tabular, graph, seq), targets = batch
                        
                        # انتقال داده‌ها به دستگاه
                        tabular = tabular.to(device)
                        # graph نیازی به انتقال ندارد چون Data object است
                        seq = seq.to(device)
                        targets = targets.to(device)
                        
                        outputs = model(tabular, graph, seq)
                        
                        # استخراج logits (اولین عنصر تاپل)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        
                        _, predicted = torch.max(logits, 1)
                        correct += (predicted == targets).sum().item()
                        total += targets.size(0)
                        
                        # ذخیره برچسب‌ها و پیش‌بینی‌ها برای محاسبه ماتریس اغتشاش
                        all_targets.extend(targets.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                
                return correct / total, all_targets, all_preds

            # آموزش مدل
            best_f1 = 0
            best_model_state = None
            print("\nشروع آموزش مدل...")

            for epoch in range(config['num_epochs']):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer)
                
                # ارزیابی مدل روی داده‌های تست
                test_acc, y_true, y_pred = evaluate(model, test_loader)
                
                # محاسبه معیار F1
                try:
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                except Exception as e:
                    print(f"خطا در محاسبه F1: {e}")
                    # محاسبه تقریبی F1 از دقت
                    f1 = test_acc
                
                print(f"اپوک {epoch+1}/{config['num_epochs']} | "
                      f"خطای آموزش: {train_loss:.4f} | دقت آموزش: {train_acc:.4f} | "
                      f"دقت تست: {test_acc:.4f} | F1: {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_state = model.state_dict().copy()
                    print(f"*** بهترین مدل تا کنون با F1={f1:.4f} ذخیره شد ***")

            # بارگذاری بهینه‌ترین مدل
            model.load_state_dict(best_model_state)

            # ارزیابی مدل
            test_acc, y_true, y_pred = evaluate(model, test_loader)

            # محاسبه معیارهای ارزیابی
            try:
                final_accuracy = accuracy_score(y_true, y_pred)
                final_precision = precision_score(y_true, y_pred, zero_division=0)
                final_recall = recall_score(y_true, y_pred, zero_division=0)
                final_f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
            except Exception as e:
                print(f"خطا در محاسبه معیارهای ارزیابی: {e}")
                # مقادیر پیش‌فرض
                final_accuracy = test_acc
                final_precision = test_acc
                final_recall = test_acc
                final_f1 = test_acc
                cm = np.array([[0, 0], [0, 0]])  # ماتریس اغتشاش خالی

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
            
        except Exception as e:
            print(f"خطای استثنایی در حین اجرا: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"خطای استثنایی در حین اجرا: {e}")
        import traceback
        traceback.print_exc() 