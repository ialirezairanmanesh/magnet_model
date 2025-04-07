import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import joblib


# بارگذاری داده‌های جدولی
X_tabular = pd.read_csv('processed_data/X_tabular.csv')
y = pd.read_csv('processed_data/y.csv')

# پیش‌پردازش داده‌ها: جایگزینی مقادیر '?' با NaN و سپس پر کردن با میانگین
X_tabular = X_tabular.replace('?', np.nan)
# تبدیل همه ستون‌ها به نوع عددی
X_tabular = X_tabular.apply(pd.to_numeric, errors='coerce')
# پر کردن مقادیر NaN با میانگین هر ستون
X_tabular = X_tabular.fillna(X_tabular.mean())

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_tabular_scaled = scaler.fit_transform(X_tabular)

# تبدیل داده‌ها به تنسور PyTorch
X_tabular_scaled = torch.FloatTensor(X_tabular_scaled)
y = torch.LongTensor(y.values.flatten())

# بررسی تعداد نمونه‌ها در هر کلاس قبل از تقسیم
unique_labels, counts = np.unique(y.numpy(), return_counts=True)
class_distribution = dict(zip(unique_labels, counts))
print("\nتوزیع اولیه کلاس‌ها:")
print(f"تعداد بدافزارها (کلاس 1): {class_distribution.get(1, 0)}")
print(f"تعداد برنامه‌های بی‌خطر (کلاس 0): {class_distribution.get(0, 0)}")

# تقسیم داده‌ها با توجه به تعداد نمونه‌ها در هر کلاس
min_samples = min(counts)
if min_samples >= 2:
    # اگر هر کلاس حداقل 2 نمونه داشته باشد، از stratify استفاده می‌کنیم
    X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
        X_tabular_scaled, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print("\nتقسیم‌بندی با حفظ نسبت کلاس‌ها انجام شد.")
else:
    # در غیر این صورت، از تقسیم‌بندی معمولی استفاده می‌کنیم
    print("\nهشدار: به دلیل کم بودن تعداد نمونه‌ها در برخی کلاس‌ها، از تقسیم‌بندی معمولی استفاده می‌شود.")
    # برای حفظ تقریبی نسبت‌ها، از shuffle استفاده می‌کنیم
    X_tabular_train, X_tabular_test, y_train, y_test = train_test_split(
        X_tabular_scaled, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )

# ذخیره داده‌های پیش‌پردازش‌شده
torch.save(X_tabular_train, 'processed_data/X_tabular_train.pt')
torch.save(X_tabular_test, 'processed_data/X_tabular_test.pt')
torch.save(y_train, 'processed_data/y_train.pt')
torch.save(y_test, 'processed_data/y_test.pt')

# ذخیره scaler برای استفاده بعدی
joblib.dump(scaler, 'processed_data/scaler.pkl')

print("\nاطلاعات تقسیم‌بندی داده‌ها:")
print(f"تعداد کل نمونه‌ها: {len(y)}")
print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
print(f"تعداد نمونه‌های آزمایشی: {len(y_test)}")

# بررسی نسبت‌ها در مجموعه آموزش
train_unique, train_counts = np.unique(y_train.numpy(), return_counts=True)
train_dist = dict(zip(train_unique, train_counts))
print("\nتوزیع کلاس‌ها در مجموعه آموزش:")
print(f"بدافزارها: {train_dist.get(1, 0)} ({train_dist.get(1, 0)/len(y_train)*100:.2f}%)")
print(f"بی‌خطرها: {train_dist.get(0, 0)} ({train_dist.get(0, 0)/len(y_train)*100:.2f}%)")

# بررسی نسبت‌ها در مجموعه آزمایش
test_unique, test_counts = np.unique(y_test.numpy(), return_counts=True)
test_dist = dict(zip(test_unique, test_counts))
print("\nتوزیع کلاس‌ها در مجموعه آزمایش:")
print(f"بدافزارها: {test_dist.get(1, 0)} ({test_dist.get(1, 0)/len(y_test)*100:.2f}%)")
print(f"بی‌خطرها: {test_dist.get(0, 0)} ({test_dist.get(0, 0)/len(y_test)*100:.2f}%)")