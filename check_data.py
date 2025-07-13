# اسکریپت بررسی نوع داده‌ها
from data_extraction import load_processed_data
import numpy as np
import pandas as pd

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

def check_dataset():
    print("Attempting to read the dataset...")
    try:
        # Try reading with pandas
        df = pd.read_csv('dataset/static_features_latest_new_4gram_latest.csv', header=None)
        print("\nDataset shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nValue counts:")
        print(df.iloc[:, -1].value_counts())
    except Exception as e:
        print("Error reading with pandas:", str(e))
        
        try:
            # Try reading with numpy
            data = np.loadtxt('dataset/static_features_latest_new_4gram_latest.csv', delimiter=',')
            print("\nDataset shape:", data.shape)
            print("\nFirst row:")
            print(data[0])
            print("\nLast column (labels):")
            unique, counts = np.unique(data[:, -1], return_counts=True)
            print(dict(zip(unique, counts)))
        except Exception as e:
            print("Error reading with numpy:", str(e))

if __name__ == '__main__':
    check_dataset() 