import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pirates import Pirates, Solution
from pirates_optimization.magnet_model import train_and_evaluate_magnet
from functions import function_factory
from magnet_objective import magnet_objective
from bcolors import bcolors
import json
from datetime import datetime
from tqdm import tqdm

# تنظیم مسیر پایتون
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# تنظیم seed برای تکرارپذیری
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# تعریف دامنه پارامترها
param_ranges = {
    'embedding_dim': (16, 128),      # بعد embedding
    'num_heads': (2, 16),            # تعداد heads در توجه چندسری
    'num_layers': (1, 4),            # تعداد لایه‌های transformer
    'dim_feedforward': (64, 512),    # بعد لایه feed-forward
    'dropout': (0.0, 0.5),           # نرخ dropout
    'batch_size': (8, 64),           # اندازه batch
    'learning_rate': (0.0001, 0.01), # نرخ یادگیری
    'weight_decay': (0.0001, 0.1),   # کاهش وزن برای تنظیم‌کننده L2
}

# تعریف تابع هدف
def objective_function(solution):
    """
    تابع هدف برای ارزیابی کیفیت یک مجموعه از پارامترها
    
    Parameters:
    -----------
    solution : Solution
        حل مساله که شامل پارامترها است
        
    Returns:
    --------
    metrics : dict
        معیارهای ارزیابی مانند دقت، صحت و امتیاز F1
    """
    try:
        # تبدیل پارامترها به مقادیر مناسب
        config = {
            'embedding_dim': int(solution.values['embedding_dim']),
            'num_heads': int(solution.values['num_heads']),
            'num_layers': int(solution.values['num_layers']),
            'dim_feedforward': int(solution.values['dim_feedforward']),
            'dropout': solution.values['dropout'],
            'batch_size': int(solution.values['batch_size']),
            'learning_rate': solution.values['learning_rate'],
            'weight_decay': solution.values['weight_decay'],
            'seq_vocab_size': 1000,  # مقادیر ثابت
            'seq_max_len': 100,
            'num_classes': 2
        }
        
        # بررسی کنید که embedding_dim بر num_heads بخش‌پذیر باشد
        if config['embedding_dim'] % config['num_heads'] != 0:
            config['embedding_dim'] = (config['embedding_dim'] // config['num_heads']) * config['num_heads']
        
        # نمایش پیشرفت
        print(f"\nارزیابی پارامترها: {config}")
        
        # ارزیابی مدل
        sample_percentage = 20  # برای سرعت بخشیدن، از 20% داده‌ها استفاده می‌کنیم
        metrics = train_and_evaluate_magnet(config, sample_percentage=sample_percentage)
        
        # نمایش نتایج
        print(f"نتایج: {metrics}")
        
        return metrics
    except Exception as e:
        import traceback
        print(f"خطا در تابع هدف: {e}")
        traceback.print_exc()
        # در صورت بروز خطا، معیارهای پایه‌ای برگردانید
        return {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5
        }

def run_pirates_optimization():
    """
    اجرای بهینه‌سازی با الگوریتم Pirates
    """
    # تعریف مساله بهینه‌سازی
    solution_template = Solution({
        'embedding_dim': ('continuous', param_ranges['embedding_dim'][0], param_ranges['embedding_dim'][1]),
        'num_heads': ('continuous', param_ranges['num_heads'][0], param_ranges['num_heads'][1]),
        'num_layers': ('continuous', param_ranges['num_layers'][0], param_ranges['num_layers'][1]),
        'dim_feedforward': ('continuous', param_ranges['dim_feedforward'][0], param_ranges['dim_feedforward'][1]),
        'dropout': ('continuous', param_ranges['dropout'][0], param_ranges['dropout'][1]),
        'batch_size': ('continuous', param_ranges['batch_size'][0], param_ranges['batch_size'][1]),
        'learning_rate': ('continuous', param_ranges['learning_rate'][0], param_ranges['learning_rate'][1]),
        'weight_decay': ('continuous', param_ranges['weight_decay'][0], param_ranges['weight_decay'][1]),
    })
    
    # تنظیم پارامترهای الگوریتم
    pirates_params = {
        'num_pirates': 10,             # تعداد دزدان دریایی (جمعیت)
        'max_iterations': 20,          # حداکثر تعداد تکرارها
        'sailing_radius': 0.3,         # شعاع جستجوی اولیه
        'plundering_radius': 0.1,      # شعاع غارت
        'strategy_change_prob': 0.2,   # احتمال تغییر استراتژی
        'max_radius_multiplier': 0.5,  # فاکتور کاهش شعاع
        'num_sailing_directions': 5,   # تعداد جهت‌های حرکت
        'sailing_length': 0.2,         # طول دریانوردی
    }
    
    # ایجاد و اجرای الگوریتم
    pirates = Pirates(solution_template, 
                      objective_function=objective_function, 
                      optimization_mode='max',
                      fitness_metric='f1',
                      **pirates_params)
    
    # اجرای بهینه‌سازی
    best_solution, best_fitness, all_fitness = pirates.run_optimization()
    
    # نمایش نتایج
    print("\n" + "="*50)
    print("بهترین پارامترها:")
    for param, value in best_solution.values.items():
        # گرد کردن مقادیر برای نمایش بهتر
        if param in ['embedding_dim', 'num_heads', 'num_layers', 'dim_feedforward', 'batch_size']:
            print(f"{param}: {int(value)}")
        else:
                        print(f"{param}: {value:.4f}")
    
    print("\nبهترین معیار F1:", best_fitness)
    print("="*50)
    
    # رسم نمودار همگرایی
    plt.figure(figsize=(10, 6))
    plt.plot(all_fitness['iteration'], all_fitness['best_fitness'], 'b-', linewidth=2)
    plt.title('Convergence Curve - Pirates Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Best F1 Score')
    plt.grid(True)
    plt.savefig('pirates_convergence.png')
    plt.show()
    
    return best_solution, best_fitness, all_fitness

if __name__ == "__main__":
    run_pirates_optimization()