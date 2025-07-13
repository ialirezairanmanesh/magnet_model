import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# بارگذاری داده‌ها
data = pd.read_csv('processed_data/X_tabular.csv')
y = pd.read_csv('processed_data/y.csv')

# --- پیش‌پردازش X ---
# تبدیل '?' به NaN و سپس به نوع عددی
data = data.replace('?', np.nan)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())  # پر کردن NaN با میانگین

# --- پیش‌پردازش y ---
# تبدیل '?' به NaN و تبدیل به نوع عددی
y = y.replace('?', np.nan)
y = y.apply(pd.to_numeric, errors='coerce')

# حذف ردیف‌هایی که y در آن‌ها NaN دارد
valid_indices = y.dropna().index
y = y.loc[valid_indices].astype(int)
data = data.loc[valid_indices]

# --- تبدیل به تنسور ---
seq_data = torch.FloatTensor(data.values)
y = torch.LongTensor(y.values.flatten())

# بررسی توزیع اولیه کلاس‌ها
print("\nتوزیع اولیه کلاس‌ها:")
unique_labels, counts = np.unique(y.numpy(), return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"کلاس {label}: {count} نمونه")

# تقسیم داده‌ها
min_samples = min(counts)
if min_samples >= 2:
    seq_train, seq_test, y_train, y_test = train_test_split(
        seq_data, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nتقسیم‌بندی با حفظ نسبت کلاس‌ها انجام شد.")
else:
    print("\nهشدار: به دلیل کم بودن تعداد نمونه‌ها در برخی کلاس‌ها، از تقسیم‌بندی معمولی استفاده می‌شود.")
    seq_train, seq_test, y_train, y_test = train_test_split(
        seq_data, y, test_size=0.2, random_state=42, shuffle=True
    )

# حذف نمونه‌های نامعتبر
def remove_invalid_samples(data, labels):
    valid_data_indices = ~torch.isnan(data).any(dim=1)
    valid_label_indices = labels >= 0
    valid_indices = valid_data_indices & valid_label_indices
    return data[valid_indices], labels[valid_indices]

seq_train, y_train = remove_invalid_samples(seq_train, y_train)
seq_test, y_test = remove_invalid_samples(seq_test, y_test)

# ذخیره داده‌ها
torch.save(seq_train, 'processed_data/seq_train.pt')
torch.save(seq_test, 'processed_data/seq_test.pt')
torch.save(y_train, 'processed_data/y_train.pt')
torch.save(y_test, 'processed_data/y_test.pt')

# گزارش نهایی
print("\nداده‌های متوالی پیش‌پردازش و ذخیره شدند.")
print(f"تعداد نمونه‌های آموزشی: {seq_train.size(0)}")
print(f"تعداد نمونه‌های آزمایشی: {seq_test.size(0)}")
print(f"طول هر توالی: {seq_train.size(1)}")

print("\nتوزیع کلاس‌ها در مجموعه آموزش:")
train_unique, train_counts = torch.unique(y_train, return_counts=True)
for label, count in zip(train_unique.tolist(), train_counts.tolist()):
    print(f"کلاس {label}: {count} نمونه ({count/len(y_train)*100:.2f}%)")

print("\nتوزیع کلاس‌ها در مجموعه آزمایش:")
test_unique, test_counts = torch.unique(y_test, return_counts=True)
for label, count in zip(test_unique.tolist(), test_counts.tolist()):
    print(f"کلاس {label}: {count} نمونه ({count/len(y_test)*100:.2f}%)")
