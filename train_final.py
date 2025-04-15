#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: آموزش نهایی مدل چندمدالیته برای پایان‌نامه
این اسکریپت مدل نهایی MAGNET را با بهینه‌ترین پارامترها آموزش می‌دهد
و مدل نهایی را برای استفاده آینده در مرحله استنتاج ذخیره می‌کند.

نویسنده: دانشجوی مقطع تحصیلات تکمیلی
تاریخ: 1402
"""

import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# وارد کردن مدل MAGNET و داده‌ها
from magnet_model import MAGNET, MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تنظیم دستگاه - استفاده از GPU در صورت وجود
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"استفاده از دستگاه: {device}")

# تنظیم حالت بازتولیدپذیری
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# تابع آموزش برای یک اپوک
def train_epoch(model, train_loader, criterion, optimizer, device, ssl_weight=0.1):
    """آموزش مدل برای یک اپوک"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_ssl_loss = 0
    correct = 0
    total = 0
    
    # حلقه آموزش با نوار پیشرفت
    with tqdm(train_loader, desc="آموزش", leave=False) as pbar:
        for batch in pbar:
            (tabular, graph, seq), targets = batch
            tabular = tabular.to(device)
            # انتقال گراف در صورت نیاز
            if hasattr(graph, 'to'):
                graph = graph.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # خروجی مدل (logits, ssl_output, attention)
            logits, ssl_output, _, _ = model(tabular, graph, seq)
            
            # محاسبه خطای طبقه‌بندی
            main_loss = criterion(logits, targets)
            
            # محاسبه خطای خودنظارتی (اگر فعال باشد)
            if ssl_weight > 0:
                # مجموع سه مدالیته به عنوان هدف خودنظارتی
                combined = torch.cat([tabular, seq], dim=1)
                
                # تطبیق ابعاد با استفاده از یک لایه خطی موقت
                if ssl_output.shape[1] != combined.shape[1]:
                    print(f"تطبیق ابعاد SSL: {ssl_output.shape} -> {combined.shape}")
                    if not hasattr(model, 'ssl_adapter'):
                        model.ssl_adapter = nn.Linear(ssl_output.shape[1], combined.shape[1]).to(device)
                    
                    # استفاده از adapter برای تبدیل ابعاد
                    adapted_output = model.ssl_adapter(ssl_output)
                    ssl_loss = nn.MSELoss()(adapted_output, combined)
                else:
                    ssl_loss = nn.MSELoss()(ssl_output, combined)
                
                loss = main_loss + ssl_weight * ssl_loss
                total_ssl_loss += ssl_loss.item()
            else:
                loss = main_loss
            
            # پس‌انتشار و بهینه‌سازی
            loss.backward()
            optimizer.step()
            
            # جمع‌آوری آمار
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            
            # محاسبه دقت
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # به‌روزرسانی نوار پیشرفت
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * correct / total:.2f}%"
            )
    
    # محاسبه مقادیر میانگین
    avg_loss = total_loss / len(train_loader)
    avg_main_loss = total_main_loss / len(train_loader)
    avg_ssl_loss = total_ssl_loss / len(train_loader) if ssl_weight > 0 else 0
    accuracy = correct / total
    
    return avg_loss, avg_main_loss, avg_ssl_loss, accuracy

# تابع ارزیابی
def evaluate(model, val_loader, criterion, device):
    """ارزیابی مدل روی داده‌های اعتبارسنجی"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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
            
            # خروجی مدل
            logits, _, _, _ = model(tabular, graph, seq)
            
            # محاسبه خطا
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # محاسبه دقت
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # ذخیره برای محاسبه معیارها
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # احتمال کلاس مثبت
            
            # به‌روزرسانی نوار پیشرفت
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * correct / total:.2f}%"
            )
    
    # محاسبه معیارهای ارزیابی
    val_loss = total_loss / len(val_loader)
    val_acc = accuracy_score(all_targets, all_preds)
    val_precision = precision_score(all_targets, all_preds, zero_division=0)
    val_recall = recall_score(all_targets, all_preds, zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # محاسبه AUC اگر هر دو کلاس موجود باشند
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except:
        val_auc = 0
    
    # ماتریس اغتشاش
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall, 
        'f1': val_f1,
        'auc': val_auc,
        'confusion_matrix': cm
    }
    
    return metrics

# تابع ذخیره مدل
def save_model(model, config, metrics, scaler, optimizer, scheduler, epoch, path):
    """ذخیره مدل و اطلاعات مرتبط با آن"""
    try:
        # تبدیل اشیاء NumPy به لیست یا عدد معمولی
        processed_metrics = {}
        for key, value in metrics.items():
            if 'numpy' in str(type(value)):
                if hasattr(value, 'tolist'):
                    processed_metrics[key] = value.tolist()
                else:
                    processed_metrics[key] = float(value)
            else:
                processed_metrics[key] = value
        
        # حذف ماتریس اغتشاش از metrics و ذخیره آن به صورت جداگانه
        if 'confusion_matrix' in processed_metrics:
            cm = processed_metrics.pop('confusion_matrix')
            if hasattr(cm, 'tolist'):
                cm = cm.tolist()
            # ذخیره ماتریس اغتشاش در یک فایل جداگانه
            cm_path = path.replace('.pth', '_confusion_matrix.npy')
            np.save(cm_path, cm)
            
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': processed_metrics,
            'epoch': epoch
        }
        
        if scaler is not None:
            save_dict['scaler'] = scaler
            
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(save_dict, path)
        return True
    except Exception as e:
        print(f"خطا در ذخیره مدل: {e}")
        # اگر ذخیره کامل شکست خورد، فقط وزن‌ها را ذخیره کنید
        try:
            print("تلاش برای ذخیره فقط وزن‌های مدل...")
            torch.save(model.state_dict(), path.replace('.pth', '_weights_only.pth'))
            return True
        except Exception as e2:
            print(f"خطا در ذخیره وزن‌های مدل: {e2}")
            return False

# تابع بارگذاری مدل
def load_model(model_path, model=None, device=None):
    """بارگذاری مدل از فایل"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # بارگذاری با weights_only=False برای پشتیبانی از PyTorch 2.6
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if model is None:
            # اگر مدل داده نشده، یک مدل جدید با پارامترهای ذخیره شده بسازید
            config = checkpoint.get('config', {})
            model = MAGNET(
                tabular_dim=config.get('tabular_dim', 215),
                graph_dim=config.get('graph_dim', 10),
                seq_dim=config.get('seq_dim', 215),
                embedding_dim=config.get('embedding_dim', 64),
                num_heads=config.get('num_heads', 2),
                num_layers=config.get('num_layers', 2),
                dim_feedforward=config.get('dim_feedforward', 256),
                dropout=config.get('dropout', 0.3),
                num_classes=config.get('num_classes', 2)
            ).to(device)
        
        # بارگذاری وزن‌های مدل
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # برگرداندن اطلاعات اضافی
        return model, checkpoint.get('scaler'), checkpoint.get('config'), checkpoint.get('results'), checkpoint.get('optimizer_state_dict'), checkpoint.get('scheduler_state_dict'), checkpoint.get('epoch', 0)
    
    except Exception as e:
        print(f"خطا در بارگذاری مدل: {e}")
        
        try:
            # تلاش مجدد با استفاده از روش جایگزین
            print("تلاش برای بارگذاری فقط وزن‌های مدل...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            if model is not None:
                model.load_state_dict(checkpoint)
                return model, None, None, None, None, None, 0
        except Exception as e2:
            print(f"خطا در بارگذاری وزن‌های مدل: {e2}")
        
        return None, None, None, None, None, None, 0

# تابع رسم نمودارها
def plot_training_curves(history, save_path=None):
    """رسم نمودارهای آموزش"""
    plt.figure(figsize=(15, 10))
    
    # نمودار خطا
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # نمودار دقت
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # نمودار F1
    plt.subplot(2, 2, 3)
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    
    # نمودار نرخ یادگیری
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"نمودارها در {save_path} ذخیره شدند")
    
    plt.show()
    
# تابع اصلی
def main(args):
    """تابع اصلی برای آموزش و ارزیابی مدل"""
    start_time = time.time()
    
    # تنظیمات پایه مدل
    config = {
        # پارامترهای معماری
        'embedding_dim': args.embedding_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        
        # پارامترهای آموزش
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.epochs,
        'alpha_ssl': args.ssl_weight,
        
        # پارامترهای داده
        'data_percentage': args.data_percentage,
        'seq_max_len': 100,
        'seq_vocab_size': 1000,
        'num_classes': 2,
    }
    
    # ایجاد دایرکتوری برای ذخیره خروجی‌ها
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/magnet_final_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # ذخیره تنظیمات
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"تنظیمات در {config_path} ذخیره شدند")
    
    # ثبت تنظیمات در خروجی
    print("\n" + "="*70)
    print(f" آموزش نهایی مدل MAGNET - {timestamp} ")
    print("="*70)
    
    print("\nتنظیمات آموزش:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # --- بارگذاری داده‌ها ---
    print("\nبارگذاری داده‌ها...")
    X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test = load_processed_data()
    
    # بعد از بارگذاری داده‌ها، اضافه کنید:
    print(f"نوع y_train: {type(y_train)}")

    # اگر y_train یک متد یا تابع است، باید آن را فراخوانی کنیم
    if callable(y_train):
        print("y_train یک تابع است، در حال فراخوانی...")
        y_train = y_train()
        print(f"نوع جدید y_train: {type(y_train)}")

    # تبدیل به numpy array برای اطمینان
    if hasattr(y_train, 'numpy'):
        y_train = y_train.numpy()  # برای تنسورهای PyTorch
    elif hasattr(y_train, 'values'):
        y_train = y_train.values  # برای DataFrame های Pandas

    y_train = np.array(y_train)
    print(f"شکل نهایی y_train: {y_train.shape}")
    
    # اطلاعات داده‌ها
    print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
    print(f"تعداد نمونه‌های آزمایشی: {len(y_test)}")
    
    # توزیع کلاس‌ها
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    train_class_dist = dict(zip(unique_train, counts_train))
    print(f"توزیع کلاس‌ها (آموزش): {train_class_dist}")
    
    # انتخاب درصدی از داده‌ها (اگر کمتر از 100% باشد)
    if args.data_percentage < 100:
        # نمونه‌گیری طبقه‌بندی شده
        train_indices_by_class = {}
        for cls in unique_train:
            train_indices_by_class[cls] = np.where(y_train == cls)[0]

        # انتخاب نمونه‌ها با حفظ نسبت کلاس‌ها
        selected_train_indices = []
        for cls, indices in train_indices_by_class.items():
            n_samples = int(len(indices) * args.data_percentage / 100)
            n_samples = max(n_samples, 10)  # حداقل 10 نمونه از هر کلاس
            selected_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
            selected_train_indices.extend(selected_indices)
        
        # ترکیب نمونه‌ها
        np.random.shuffle(selected_train_indices)
        
        # انتخاب داده‌ها
        X_tabular_train_subset = X_tabular_train[selected_train_indices]
        seq_data_train_subset = seq_data_train[selected_train_indices]
        y_train_subset = y_train[selected_train_indices]
        
        print(f"استفاده از {args.data_percentage}% داده‌ها: {len(y_train_subset)} نمونه آموزشی")
    else:
        X_tabular_train_subset = X_tabular_train
        seq_data_train_subset = seq_data_train
        y_train_subset = y_train
        print(f"استفاده از 100% داده‌ها: {len(y_train)} نمونه آموزشی")
    
    # --- پیش‌پردازش داده‌ها ---
    print("\nپیش‌پردازش و استانداردسازی داده‌ها...")
    scaler = StandardScaler()
    X_tabular_train_scaled = scaler.fit_transform(X_tabular_train_subset)
    X_tabular_test_scaled = scaler.transform(X_tabular_test)
    
    # --- ایجاد دیتاست‌ها ---
    print("ایجاد دیتاست‌ها و دیتالودرها...")
    train_dataset = MultiModalDataset(X_tabular_train_scaled, graph_data, seq_data_train_subset, y_train_subset)
    test_dataset = MultiModalDataset(X_tabular_test_scaled, graph_data, seq_data_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers
    )
    
    # --- آماده‌سازی مدل ---
    print("\nآماده‌سازی مدل MAGNET...")
    tabular_dim = X_tabular_train.shape[1]
    graph_node_dim = graph_data.x.shape[1]
    graph_edge_dim = graph_data.edge_attr.shape[1] if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else 0
    
    model = MAGNET(
        tabular_dim=tabular_dim,
        graph_node_dim=graph_node_dim,
        graph_edge_dim=graph_edge_dim,
        seq_vocab_size=config['seq_vocab_size'],
        seq_max_len=config['seq_max_len'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_classes=config['num_classes']
    )
    
    model.to(device)
    
    # محاسبه تعداد پارامترها
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"تعداد کل پارامترها: {total_params:,}")
    print(f"تعداد پارامترهای قابل آموزش: {trainable_params:,}")
    
    # --- تنظیم تابع ضرر و بهینه‌ساز ---
    # محاسبه وزن کلاس‌ها برای داده‌های نامتوازن
    if args.class_weights:
        # محاسبه وزن کلاس‌ها برای داده‌های نامتوازن
        class_counts = np.bincount(y_train_subset.astype(int))
        total_samples = len(y_train_subset)
        class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
        class_weights = class_weights.to(device)
        print(f"وزن کلاس‌ها: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # برنامه‌ریز نرخ یادگیری: کاهش در صورت عدم بهبود
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # --- آموزش مدل ---
    print("\n" + "="*50)
    print(f" شروع آموزش مدل ")
    print("="*50)
    
    best_val_f1 = 0
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = args.patience
    
    # متغیرهایی برای ثبت سابقه آموزش
    history = {
        'train_loss': [],
        'train_main_loss': [],
        'train_ssl_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'learning_rate': []
    }
    
    # لوپ اصلی آموزش
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # آموزش یک اپوک
        train_loss, train_main_loss, train_ssl_loss, train_acc = train_epoch(
            model, 
            train_loader, 
            criterion,
            optimizer, 
            device,
            ssl_weight=config['alpha_ssl']
        )
        
        # ارزیابی مدل
        val_metrics = evaluate(model, test_loader, criterion, device)
        
        # ثبت نرخ یادگیری
        current_lr = optimizer.param_groups[0]['lr']
        
        # به‌روزرسانی برنامه‌ریز نرخ یادگیری
        scheduler.step(val_metrics['f1'])
        
        # ثبت سابقه
        history['train_loss'].append(train_loss)
        history['train_main_loss'].append(train_main_loss)
        history['train_ssl_loss'].append(train_ssl_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['learning_rate'].append(current_lr)
        
        # نمایش نتایج
        print(f"زمان اپوک: {time.time() - epoch_start_time:.2f} ثانیه")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"LR: {current_lr}")
        
        # بررسی ماتریس اغتشاش
        print("ماتریس اغتشاش:")
        print(val_metrics['confusion_matrix'])
        
        # ذخیره بهترین مدل بر اساس F1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            early_stop_counter = 0
            
            # ذخیره بهترین مدل
            best_model_path = os.path.join(output_dir, f"magnet_best_model.pth")
            save_model(
                model,
                config,
                val_metrics,
                scaler,
                optimizer,
                scheduler,
                epoch,
                best_model_path
            )
            print(f"*** بهترین مدل با F1={best_val_f1:.4f} ذخیره شد ***")
        else:
            early_stop_counter += 1
            
        # بررسی توقف زودهنگام
        if early_stop_counter >= early_stop_patience:
            print(f"\nتوقف زودهنگام! بدون بهبود در {early_stop_patience} اپوک اخیر.")
            break
            
    # --- ارزیابی نهایی ---
    print("\n" + "="*50)
    print(f" ارزیابی نهایی مدل ")
    print("="*50)
    
    # بارگذاری بهترین مدل
    best_model_path = os.path.join(output_dir, f"magnet_best_model.pth")
    model, _, _, _, _, _, _ = load_model(best_model_path, model)
    
    # ارزیابی نهایی
    final_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nنتایج نهایی (بهترین مدل از اپوک {best_epoch+1}):")
    print(f"دقت: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"AUC: {final_metrics['auc']:.4f}")
    
    print("\nماتریس اغتشاش نهایی:")
    print(final_metrics['confusion_matrix'])
    
    # ذخیره نمودارها
    plot_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(history, save_path=plot_path)
    
    # ذخیره تاریخچه آموزش
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        # تبدیل مقادیر numpy به لیست
        history_serializable = {}
        for key, value in history.items():
            history_serializable[key] = [float(v) for v in value]
        json.dump(history_serializable, f, indent=4)
    
    # زمان اجرا
    total_time = time.time() - start_time
    print(f"\nزمان کل اجرا: {timedelta(seconds=total_time)}")
    print(f"مدل نهایی و خروجی‌ها در {output_dir} ذخیره شدند")
    
    # ذخیره مدل نهایی (معمولاً همان بهترین مدل است)
    final_model_path = os.path.join(output_dir, f"magnet_final_model.pth")
    save_model(
        model,
        config,
        final_metrics,
        scaler,
        optimizer,
        scheduler,
        config['num_epochs']-1,
        final_model_path
    )
    
    return model, final_metrics, history, output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='آموزش نهایی مدل MAGNET برای پایان‌نامه')
    
    # پارامترهای معماری
    parser.add_argument('--embedding_dim', type=int, default=64, 
                        help='ابعاد embedding (پیشنهاد: 64)')
    parser.add_argument('--num_heads', type=int, default=2, 
                        help='تعداد سرهای توجه در ترانسفورمر (پیشنهاد: 2)')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='تعداد لایه‌های ترانسفورمر (پیشنهاد: 2)')
    parser.add_argument('--dim_feedforward', type=int, default=256, 
                        help='ابعاد لایه feed-forward در ترانسفورمر (پیشنهاد: 256)')
    parser.add_argument('--dropout', type=float, default=0.3, 
                        help='نرخ dropout (پیشنهاد: 0.3)')
    
    # پارامترهای آموزش
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='اندازه batch (پیشنهاد: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                        help='نرخ یادگیری (پیشنهاد: 0.0005)')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='ضریب weight decay (پیشنهاد: 0.01)')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='تعداد اپوک‌ها (پیشنهاد: 30)')
    parser.add_argument('--ssl_weight', type=float, default=0, 
                        help='ضریب وزن یادگیری خودنظارتی (پیشنهاد: 0)')
    
    # پارامترهای اجرا
    parser.add_argument('--data_percentage', type=int, default=50, 
                        help='درصد داده‌ها برای آموزش (پیشنهاد: 50)')
    parser.add_argument('--patience', type=int, default=7, 
                        help='تعداد اپوک برای early stopping (پیشنهاد: 7)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='تعداد کارگرها برای dataloader (پیشنهاد: 4)')
    parser.add_argument('--class_weights', action='store_true', 
                        help='استفاده از وزن‌دهی به کلاس‌ها برای داده‌های نامتوازن')
    
    args = parser.parse_args()
    
    # اجرای تابع اصلی
    main(args) 