import torch
from sklearn.model_selection import train_test_split
import numpy as np

# بارگذاری داده‌های آموزشی اولیه
X_tabular_train = torch.load('processed_data/X_tabular_train.pt')
seq_train = torch.load('processed_data/seq_train.pt')
y_train = torch.load('processed_data/y_train.pt')

# چاپ اطلاعات اولیه برای دیباگ
print(f"ابعاد اولیه - X_tabular_train: {X_tabular_train.shape}, seq_train: {seq_train.shape}, y_train: {y_train.shape}")

# بررسی و تطبیق تعداد نمونه‌ها
min_samples = min(X_tabular_train.shape[0], seq_train.shape[0], y_train.shape[0])
if X_tabular_train.shape[0] != min_samples or seq_train.shape[0] != min_samples or y_train.shape[0] != min_samples:
    print(f"تعداد نمونه‌ها متفاوت است. برش داده‌ها به {min_samples} نمونه...")
    X_tabular_train = X_tabular_train[:min_samples]
    seq_train = seq_train[:min_samples]
    y_train = y_train[:min_samples]

# پاکسازی برچسب‌های عجیب
mask = (y_train == 0) | (y_train == 1)
X_tabular_train = X_tabular_train[mask]
seq_train = seq_train[mask]
y_train = y_train[mask]

# تقسیم داده‌ها به آموزشی و اعتبارسنجی (80-20)
X_tabular_train_new, X_tabular_val, seq_train_new, seq_val, y_train_new, y_val = train_test_split(
    X_tabular_train, seq_train, y_train,
    test_size=0.2, random_state=42, stratify=y_train
)

# ذخیره داده‌های جدید
torch.save(X_tabular_train_new, 'processed_data/X_tabular_train.pt')
torch.save(X_tabular_val, 'processed_data/X_tabular_val.pt')
torch.save(seq_train_new, 'processed_data/seq_train.pt')
torch.save(seq_val, 'processed_data/seq_val.pt')
torch.save(y_train_new, 'processed_data/y_train.pt')
torch.save(y_val, 'processed_data/y_val.pt')

print("\nداده‌های آموزشی و اعتبارسنجی با موفقیت ذخیره شدند.")
print(f"تعداد نمونه‌های آموزشی: {len(y_train_new)}")
print(f"تعداد نمونه‌های اعتبارسنجی: {len(y_val)}")
print(f"توزیع کلاس‌ها در داده‌های آموزشی: {torch.bincount(y_train_new)}")
print(f"توزیع کلاس‌ها در داده‌های اعتبارسنجی: {torch.bincount(y_val)}")