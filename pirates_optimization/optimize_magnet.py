import numpy as np
from pirates import Pirates
from functions import function_factory
from magnet_objective import magnet_objective
from bcolors import bcolors
import json
from datetime import datetime
import os
from tqdm import tqdm

def optimize_magnet():
    """
    بهینه‌سازی پارامترهای مدل MAGNET با استفاده از الگوریتم Pirates
    """
    # تعریف محدوده پارامترها - اصلاح شده برای اطمینان از بخش‌پذیری
    param_bounds = {
        'embedding_dim': (64, 256),    # بعد embedding - شروع از 64 برای بخش‌پذیری بر اعداد کوچک
        'num_heads': (2, 8),           # تعداد headهای attention - محدود به 8 برای بخش‌پذیری بهتر
        'num_layers': (1, 4),          # تعداد لایه‌های transformer
        'dim_feedforward': (128, 512),  # بعد لایه feedforward
        'dropout': (0.1, 0.5)          # نرخ dropout
    }
    
    # تبدیل محدوده‌ها به فرمت مورد نیاز الگوریتم
    fmin = np.array([v[0] for v in param_bounds.values()])
    fmax = np.array([v[1] for v in param_bounds.values()])
    
    # تنظیم پارامترهای الگوریتم
    num_ships = 20          # تعداد کشتی‌ها
    max_iter = 30          # کاهش تعداد تکرارها
    top_ships = 5          # تعداد کشتی‌های برتر
    max_wind = 0.5         # حداکثر سرعت باد
    max_r = 0.5           # حداکثر شعاع جستجو
    p_hr = 0.1            # احتمال وقوع طوفان
    
    # وزن‌های به‌روزرسانی سرعت
    c = {
        'leader': 0.5,          # وزن رهبر
        'private_map': 0.5,     # وزن نقشه خصوصی
        'map': 0.5,             # وزن نقشه عمومی
        'top_ships': 0.5        # وزن کشتی‌های برتر
    }
    
    print(f"{bcolors.HEADER}شروع بهینه‌سازی پارامترهای مدل MAGNET{bcolors.ENDC}")
    print(f"{bcolors.BLUE}محدوده پارامترها:{bcolors.ENDC}")
    for param, (min_val, max_val) in param_bounds.items():
        print(f"{param}: [{min_val}, {max_val}]")
    
    # تعریف تابع هدف
    class MagnetFunction:
        def __init__(self):
            self.dim = len(param_bounds)
            self.fmin = fmin
            self.fmax = fmax
            self.func = magnet_objective
            
        def __call__(self, x):
            # اطمینان از بخش‌پذیری embedding_dim بر num_heads
            embedding_dim = int(x[0])
            num_heads = int(x[1])
            # گرد کردن embedding_dim به نزدیک‌ترین مضرب num_heads
            embedding_dim = (embedding_dim // num_heads) * num_heads
            x[0] = embedding_dim
            return self.func(x)
    
    # ایجاد نمونه از الگوریتم
    pirates = Pirates(
        func=MagnetFunction(),
        fmin=fmin,
        fmax=fmax,
        num_ships=num_ships,
        dimensions=len(param_bounds),
        max_iter=max_iter,
        top_ships=top_ships,
        max_wind=max_wind,
        max_r=max_r,
        hr=p_hr,
        c=c
    )
    
    # ذخیره نتایج
    results = {
        'best_params': None,
        'best_error': float('inf'),
        'history': [],
        'config': {
            'param_bounds': param_bounds,
            'algorithm_params': {
                'num_ships': num_ships,
                'max_iter': max_iter,
                'top_ships': top_ships,
                'max_wind': max_wind,
                'max_r': max_r,
                'p_hr': p_hr,
                'c': c
            }
        }
    }
    
    # پارامترهای توقف زودهنگام
    early_stop_patience = 5  # تعداد تکرارهای بدون بهبود
    no_improve_iterations = 0
    min_error_delta = 0.001  # حداقل بهبود مورد نیاز
    
    # محاسبه کل تعداد ارزیابی‌ها
    total_evaluations = max_iter * num_ships
    current_evaluation = 0
    
    # اجرای الگوریتم با نوار پیشرفت
    print(f"\n{bcolors.BLUE}شروع اجرای الگوریتم Pirates...{bcolors.ENDC}")
    print(f"تعداد کل ارزیابی‌ها: {total_evaluations}")
    
    with tqdm(total=total_evaluations, desc="پیشرفت کلی", position=0) as pbar_main:
        with tqdm(total=max_iter, desc="تکرارها", position=1, leave=False) as pbar_iter:
            for i in range(max_iter):
                # اجرای یک تکرار
                error, params = pirates.run_iteration()
                current_evaluation += num_ships
                
                # ذخیره نتایج
                results['history'].append({
                    'iteration': i + 1,
                    'error': error,
                    'params': dict(zip(param_bounds.keys(), params))
                })
                
                # به‌روزرسانی بهترین نتایج
                if error < results['best_error'] - min_error_delta:
                    results['best_error'] = error
                    results['best_params'] = dict(zip(param_bounds.keys(), params))
                    no_improve_iterations = 0
                    
                    print(f"\n{bcolors.GREEN}بهترین نتیجه جدید در تکرار {i+1}:{bcolors.ENDC}")
                    print(f"خطا: {error:.6f}")
                    print("پارامترها:")
                    for param, value in results['best_params'].items():
                        print(f"{param}: {value:.4f}")
                else:
                    no_improve_iterations += 1
                
                # بررسی توقف زودهنگام
                if no_improve_iterations >= early_stop_patience:
                    print(f"\n{bcolors.YELLOW}توقف زودهنگام: عدم بهبود در {early_stop_patience} تکرار متوالی{bcolors.ENDC}")
                    break
                
                # به‌روزرسانی نوار پیشرفت
                pbar_iter.update(1)
                pbar_main.update(num_ships)
                pbar_main.set_postfix({
                    'best_error': f"{results['best_error']:.6f}",
                    'progress': f"{(current_evaluation/total_evaluations*100):.1f}%"
                })
    
    # ذخیره نتایج در فایل
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results/pirates_optimization"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"pirates_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{bcolors.GREEN}بهینه‌سازی با موفقیت به پایان رسید!{bcolors.ENDC}")
    print(f"نتایج در فایل {results_file} ذخیره شد.")
    print("\nبهترین پارامترهای یافت شده:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value:.4f}")
    print(f"خطای نهایی: {results['best_error']:.6f}")

if __name__ == "__main__":
    optimize_magnet()