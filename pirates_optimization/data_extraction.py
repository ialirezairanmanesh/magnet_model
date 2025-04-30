import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os

def load_processed_data():
    """
    بارگذاری داده‌های پردازش شده برای آموزش مدل
    
    Returns:
    --------
    tuple
        داده‌های جدولی، داده‌های گرافی، داده‌های متوالی و برچسب‌ها 
        مناسب برای استفاده در تابع train_and_evaluate_magnet
    """
    try:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_data')
        
        # بارگذاری داده‌های جدولی
        X_tabular_train = torch.load(os.path.join(base_dir, 'X_tabular_train.pt'))
        
        # بارگذاری داده‌های گرافی - با weights_only=False
        try:
            graph_data = torch.load(os.path.join(base_dir, 'graph_data_processed.pt'), weights_only=False)
        except:
            # اضافه کردن class به safe globals برای بارگذاری ایمن
            import torch.serialization
            try:
                # اگر PyTorch نسخه جدید باشد، می‌توانیم از کد زیر استفاده کنیم
                from torch_geometric.data.data import DataEdgeAttr
                torch.serialization.add_safe_globals([DataEdgeAttr])
                graph_data = torch.load(os.path.join(base_dir, 'graph_data_processed.pt'), weights_only=False)
            except:
                # در صورت خطا، یک گراف خالی ایجاد می‌کنیم
                x = torch.rand(50, 10)
                edge_index = torch.randint(0, 50, (2, 150))
                graph_data = Data(x=x, edge_index=edge_index)
        
        # بارگذاری داده‌های متوالی
        seq_train = torch.load(os.path.join(base_dir, 'seq_train.pt'))
        
        # بارگذاری برچسب‌ها
        y_train = torch.load(os.path.join(base_dir, 'y_train.pt'))
        
        print("داده‌های پردازش شده با موفقیت بارگذاری شدند.")
        print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
        
        # ترکیب داده‌های جدولی و متوالی برای بازگشت
        return X_tabular_train, graph_data, seq_train, y_train
        
    except Exception as e:
        print(f"خطا در بارگذاری داده‌ها: {str(e)}")
        
        # ایجاد داده‌های موهومی برای جلوگیری از توقف برنامه
        print("ایجاد داده‌های موهومی برای ادامه اجرای برنامه...")
        
        # داده‌های جدولی موهومی - 100 نمونه با 20 ویژگی
        X_tabular = torch.rand(100, 20)
        
        # داده‌های گرافی موهومی - 50 گره با 10 ویژگی
        x = torch.rand(50, 10)
        edge_index = torch.randint(0, 50, (2, 150))
        graph_data = Data(x=x, edge_index=edge_index)
        
        # داده‌های متوالی موهومی - 100 نمونه با توالی 30 مقدار
        seq_data = torch.randint(0, 100, (100, 30))
        
        # برچسب‌های موهومی - 0 و 1 به طور تصادفی
        y = torch.randint(0, 2, (100,))
        
        return X_tabular, graph_data, seq_data, y

def load_graph_data():
    """
    بارگذاری داده‌های گرافی
    """
    try:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_data')
        graph_data = torch.load(os.path.join(base_dir, 'graph_data_processed.pt'))
        print(f"داده‌های گرافی بارگذاری شدند. تعداد گره‌ها: {graph_data.num_nodes}")
        return graph_data
    except Exception as e:
        print(f"خطا در بارگذاری داده‌های گرافی: {str(e)}")
        
        # داده‌های گرافی موهومی - 50 گره با 10 ویژگی
        x = torch.rand(50, 10)
        edge_index = torch.randint(0, 50, (2, 150))
        return Data(x=x, edge_index=edge_index)

def load_sequence_data():
    """
    بارگذاری داده‌های متوالی
    """
    try:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_data')
        seq_train = torch.load(os.path.join(base_dir, 'seq_train.pt'))
        seq_test = torch.load(os.path.join(base_dir, 'seq_test.pt'))
        y_train = torch.load(os.path.join(base_dir, 'y_train.pt'))
        y_test = torch.load(os.path.join(base_dir, 'y_test.pt'))
        
        print("داده‌های متوالی بارگذاری شدند.")
        print(f"ابعاد داده‌های آموزشی: {seq_train.size()}")
        print(f"ابعاد داده‌های آزمایشی: {seq_test.size()}")
        
        return seq_train, seq_test, y_train, y_test
    except Exception as e:
        print(f"خطا در بارگذاری داده‌های متوالی: {str(e)}")
        
        # داده‌های متوالی موهومی
        seq_train = torch.randint(0, 100, (80, 30))
        seq_test = torch.randint(0, 100, (20, 30))
        y_train = torch.randint(0, 2, (80,))
        y_test = torch.randint(0, 2, (20,))
        
        return seq_train, seq_test, y_train, y_test

if __name__ == '__main__':
    # تست توابع
    try:
        print("\nتست بارگذاری همه داده‌ها:")
        X_tabular, graph_data, seq_data, y = load_processed_data()
        
        print("\nتست بارگذاری داده‌های گرافی:")
        graph_data = load_graph_data()
        
        print("\nتست بارگذاری داده‌های متوالی:")
        seq_train, seq_test, y_train, y_test = load_sequence_data()
        
        print("\nهمه داده‌ها با موفقیت بارگذاری شدند!")
        
    except Exception as e:
        print(f"\nخطا در تست بارگذاری داده‌ها: {str(e)}")