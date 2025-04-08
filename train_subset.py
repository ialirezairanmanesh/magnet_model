"""
آموزش مدل MAGNET روی درصدی از داده‌ها با قابلیت توقف زودهنگام و بهینه‌سازی بیشتر
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from datetime import timedelta
import argparse
import os

# وارد کردن مدل و توابع مورد نیاز
from magnet_model import MAGNET, train_and_evaluate_magnet
from sklearn.utils.class_weight import compute_class_weight

if __name__ == '__main__':
    # پارامترهای خط فرمان برای انعطاف‌پذیری بیشتر
    parser = argparse.ArgumentParser(description='آموزش مدل MAGNET روی درصدی از داده‌ها')
    parser.add_argument('--percentage', type=int, default=10, 
                        help='درصد داده‌ها برای آموزش (عدد بین 1 تا 100)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='تعداد اپوک‌ها برای آموزش')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='اندازه بچ برای آموزش')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='نرخ یادگیری')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='توقف زودهنگام پس از چند اپوک بدون بهبود')
    parser.add_argument('--data_path', type=str, default='processed_data',
                        help='مسیر داده‌های پردازش شده')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f" آموزش مدل MAGNET روی {args.percentage}% از داده‌ها ")
    print(f"{'='*70}")
    
    # تنظیمات مدل بهبود یافته
    config = {
        # پارامترهای مدل - افزایش ابعاد
        'embedding_dim': 128,
        'num_heads': 8,
        'num_layers': 6,
        'dim_feedforward': 512,
        'dropout': 0.3,

        # پارامترهای آموزش - تنظیم بهتر
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'early_stop_patience': args.early_stop,
        'grad_accumulation_steps': 4,  # تجمیع گرادیان برای بچ‌های بزرگتر موثر
        'alpha_ssl': 0.1,  # کاهش وزن SSL برای تمرکز بیشتر روی طبقه‌بندی

        # پارامترهای داده
        'seq_max_len': 100,
        'seq_vocab_size': 1000,  # اطمینان حاصل کنید که این مقدار برای داده‌های واقعی درست باشد
        'num_classes': 2,
        
        # مسیر داده‌ها
        'data_path': args.data_path,
    }
    
    # ایجاد دایرکتوری برای ذخیره مدل‌ها اگر وجود ندارد
    os.makedirs('models', exist_ok=True)
    
    # نمایش پارامترهای کلیدی
    print(f"\nپارامترهای آموزش:")
    print(f"  درصد داده‌ها: {args.percentage}%")
    print(f"  تعداد اپوک‌ها: {config['num_epochs']}")
    print(f"  اندازه بچ: {config['batch_size']} (با تجمیع گرادیان: {config['batch_size']*config['grad_accumulation_steps']})")
    print(f"  نرخ یادگیری: {config['learning_rate']}")
    print(f"  توقف زودهنگام پس از {config['early_stop_patience']} اپوک بدون بهبود")
    print(f"  مسیر داده‌ها: {config['data_path']}")
    
    # اندازه‌گیری زمان اجرا
    start_time = time.time()
    
    # فراخوانی تابع آموزش با درصد مشخص‌شده
    print(f"\nشروع آموزش مدل...")
    model, scaler, results = train_and_evaluate_magnet(config, args.percentage)
    
    # محاسبه و نمایش زمان اجرا
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nزمان اجرا: {timedelta(seconds=elapsed_time)}")
    
    # نمایش نتایج نهایی
    print(f"\nنتایج نهایی:")
    print(f"  دقت: {results['Accuracy']:.4f}")
    print(f"  صحت: {results['Precision']:.4f}")
    print(f"  فراخوانی: {results['Recall']:.4f}")
    print(f"  معیار F1: {results['F1 Score']:.4f}")
    
    # ذخیره مدل با نام حاوی درصد داده‌ها و معیار F1
    f1_score = results['F1 Score']
    model_path = f'models/magnet_{args.percentage}percent_f1_{f1_score:.4f}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': config,
        'results': results
    }, model_path)
    
    print(f"\nمدل با موفقیت در {model_path} ذخیره شد.")

    # در magnet_model.py، بهبود محاسبه وزن‌های کلاس
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.FloatTensor(class_weights).to(device) 