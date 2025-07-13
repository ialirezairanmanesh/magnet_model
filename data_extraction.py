import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def load_processed_data():
    """
    بارگذاری داده‌های پردازش شده برای آموزش مدل
    """
    try:
        # بارگذاری داده‌های جدولی
        X_tabular_train = torch.load('processed_data/X_tabular_train.pt')
        X_tabular_test = torch.load('processed_data/X_tabular_test.pt')
        
        # بارگذاری داده‌های گرافی
        graph_data = torch.load('processed_data/graph_data_processed.pt', weights_only=False)
        
        # بارگذاری داده‌های متوالی
        seq_data_train = torch.load('processed_data/seq_train.pt')
        seq_data_test = torch.load('processed_data/seq_test.pt')
        
        # بارگذاری برچسب‌ها
        y_train = torch.load('processed_data/y_train.pt')
        y_test = torch.load('processed_data/y_test.pt')
        
        print("داده‌های پردازش شده با موفقیت بارگذاری شدند.")
        print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
        print(f"تعداد نمونه‌های آزمایشی: {len(y_test)}")
        
        return X_tabular_train, X_tabular_test, graph_data, seq_data_train, seq_data_test, y_train, y_test
        
    except Exception as e:
        print(f"خطا در بارگذاری داده‌ها: {str(e)}")
        raise

def load_graph_data():
    """
    بارگذاری داده‌های گرافی
    """
    try:
        graph_data = torch.load('processed_data/graph_data_processed.pt', weights_only=False)
        print(f"داده‌های گرافی بارگذاری شدند. تعداد گره‌ها: {graph_data.num_nodes}")
        return graph_data
    except Exception as e:
        print(f"خطا در بارگذاری داده‌های گرافی: {str(e)}")
        raise

def load_sequence_data():
    """
    بارگذاری داده‌های متوالی
    """
    try:
        seq_train = torch.load('processed_data/seq_train.pt')
        seq_test = torch.load('processed_data/seq_test.pt')
        y_train = torch.load('processed_data/y_train.pt')
        y_test = torch.load('processed_data/y_test.pt')
        
        print("داده‌های متوالی بارگذاری شدند.")
        print(f"ابعاد داده‌های آموزشی: {seq_train.size()}")
        print(f"ابعاد داده‌های آزمایشی: {seq_test.size()}")
        
        return seq_train, seq_test, y_train, y_test
    except Exception as e:
        print(f"خطا در بارگذاری داده‌های متوالی: {str(e)}")
        raise

if __name__ == '__main__':
    # تست توابع
    try:
        print("\nتست بارگذاری همه داده‌ها:")
        X_tabular_train, X_tabular_test, graph_data, seq_train, seq_test, y_train, y_test = load_processed_data()
        
        print("\nتست بارگذاری داده‌های گرافی:")
        graph_data = load_graph_data()
        
        print("\nتست بارگذاری داده‌های متوالی:")
        seq_train, seq_test, y_train, y_test = load_sequence_data()
        
        print("\nهمه داده‌ها با موفقیت بارگذاری شدند!")
        
    except Exception as e:
        print(f"\nخطا در تست بارگذاری داده‌ها: {str(e)}")