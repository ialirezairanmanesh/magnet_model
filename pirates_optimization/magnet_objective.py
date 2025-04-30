import torch
import numpy as np
from magnet_model import MAGNET, train_and_evaluate_magnet
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def magnet_objective(x):
    """
    تابع هدف برای بهینه‌سازی پارامترهای مدل MAGNET
    
    Parameters:
    -----------
    x : numpy.ndarray
        آرایه پارامترهای بهینه‌سازی شامل:
        [embedding_dim, num_heads, num_layers, dim_feedforward, dropout]
    
    Returns:
    --------
    float
        خطای مدل (هرچه کمتر بهتر)
    """
    print("\n" + "="*50)
    print("شروع ارزیابی مجموعه پارامترهای جدید")
    print("="*50)
    
    # تبدیل پارامترها به مقادیر مناسب
    embedding_dim = int(x[0])  # بعد embedding
    num_heads = int(x[1])      # تعداد headهای attention
    num_layers = int(x[2])     # تعداد لایه‌های transformer
    dim_feedforward = int(x[3]) # بعد لایه feedforward
    dropout = x[4]             # نرخ dropout
    
    print(f"\nپارامترهای بهینه‌سازی:")
    print(f"embedding_dim: {embedding_dim}")
    print(f"num_heads: {num_heads}")
    print(f"num_layers: {num_layers}")
    print(f"dim_feedforward: {dim_feedforward}")
    print(f"dropout: {dropout:.3f}")
    
    # تنظیم پارامترهای مدل
    config = {
        # پارامترهای بهینه‌سازی شده
        'embedding_dim': embedding_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'dropout': dropout,
        
        # پارامترهای ثابت
        'seq_vocab_size': 1000,  # اندازه دیکشنری توکن‌ها
        'seq_max_len': 100,      # حداکثر طول دنباله
        'num_classes': 2,        # تعداد کلاس‌ها
        
        # پارامترهای آموزش
        'batch_size': 64,
        'num_epochs': 5,         # کاهش تعداد اپوک‌ها
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        
        # پارامترهای اضافی
        'fast_mode': True,       # فعال کردن حالت سریع
        'skip_validation': True, # رد کردن validation برای سرعت بیشتر
        'validation_frequency': 5,
        'grad_accumulation_steps': 1,
        
        # پارامترهای توقف زودهنگام
        'early_stop_patience': 5  # توقف زودهنگام بعد از 3 اپوک بدون بهبود
    }
    
    print("\nآماده‌سازی برای آموزش مدل...")
    print(f"تعداد اپوک: {config['num_epochs']}")
    print(f"اندازه دسته: {config['batch_size']}")
    print(f"درصد داده‌های استفاده شده: 50%")
    
    try:
        print("\nشروع آموزش و ارزیابی مدل...")
        
        # نوار پیشرفت برای ارزیابی پارامترها
        with tqdm(total=config['num_epochs'], desc="پیشرفت ارزیابی", position=2) as pbar_eval:
            def update_progress(epoch, loss, metrics):
                pbar_eval.update(1)
                pbar_eval.set_postfix({
                    'epoch': epoch + 1,
                    'loss': f"{loss:.4f}",
                    'f1': f"{metrics['f1']:.4f}" if metrics else 'N/A'
                })
            
            metrics = train_and_evaluate_magnet(config, sample_percentage=20, progress_callback=update_progress)
        
        print("\nنتایج ارزیابی:")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # استفاده از F1-score به عنوان معیار بهینه‌سازی
        error = 1 - metrics['f1']  # تبدیل به خطا (هرچه کمتر بهتر)
        print(f"\nخطای محاسبه شده: {error:.4f}")
        
        print("\n" + "="*50)
        print("پایان ارزیابی این مجموعه پارامترها")
        print("="*50 + "\n")
        
        return error, metrics  # برگرداندن خطا و تمام metrics
    
    except Exception as e:
        # در صورت بروز خطا، یک مقدار بزرگ برگردان
        print(f"\nخطا در ارزیابی: {str(e)}")
        print("برگرداندن حداکثر خطا (1.0)")
        return 1.0, {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}  # برگرداندن خطا و metrics خالی 