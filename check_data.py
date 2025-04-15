# اسکریپت بررسی نوع داده‌ها
from data_extraction import load_processed_data
import numpy as np

data = load_processed_data()
print("نوع داده برگشتی:", type(data))

if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: نوع={type(value)}, ", end="")
        if callable(value):
            print("(تابع است)")
        elif hasattr(value, 'shape'):
            print(f"شکل={value.shape}")
        else:
            print("")
elif isinstance(data, tuple):
    for i, item in enumerate(data):
        print(f"آیتم {i}: نوع={type(item)}, ", end="")
        if callable(item):
            print("(تابع است)")
        elif hasattr(item, 'shape'):
            print(f"شکل={item.shape}")
        else:
            print("") 