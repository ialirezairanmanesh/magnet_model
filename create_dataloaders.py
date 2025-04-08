import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import json

try:
    # Load data
    X_tabular_train = torch.load('processed_data/X_tabular_train.pt')
    X_tabular_test = torch.load('processed_data/X_tabular_test.pt')
    graph_data = torch.load('processed_data/graph_data_processed.pt', weights_only=False)
    seq_train = torch.load('processed_data/seq_train.pt')
    seq_test = torch.load('processed_data/seq_test.pt')
    y_train = torch.load('processed_data/y_train.pt')
    y_test = torch.load('processed_data/y_test.pt')

    # Load best hyperparameters
    with open('processed_data/best_config.json', 'r') as f:
        config = json.load(f)
    
    print("بهترین هایپرپارامترها:")
    print(config)

    # Check class distribution
    unique_classes, class_counts = torch.unique(y_train, return_counts=True)
    print("\nتوزیع کلاس‌ها در داده‌های آموزشی:")
    for cls, count in zip(unique_classes.tolist(), class_counts.tolist()):
        print(f"کلاس {cls}: {count} نمونه")

    # نگاشت برچسب‌های منفی به مقادیر غیر منفی
    unique_labels = torch.unique(y_train)
    label_map = {label.item(): 1 if label.item() > 0 else 0 for label in unique_labels}  # تغییر نگاشت به فقط 0 و 1

    print("\nنگاشت برچسب‌ها:")
    for original, mapped in label_map.items():
        print(f"برچسب اصلی {original} -> برچسب جدید {mapped}")

    # تبدیل برچسب‌ها
    y_train = torch.tensor([label_map[label.item()] for label in y_train])
    y_test = torch.tensor([label_map[label.item()] for label in y_test])

    # جدا کردن مجموعه اعتبارسنجی
    X_tabular_train, X_tabular_val, seq_train, seq_val, y_train, y_val = train_test_split(
        X_tabular_train, seq_train, y_train, 
        test_size=0.2, 
        random_state=42
    )

    class MultiModalDataset(Dataset):
        def __init__(self, X_tabular, graph_data, seq_data, y):
            self.X_tabular = X_tabular
            self.graph_data = graph_data
            self.seq_data = seq_data
            self.y = y
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return (self.X_tabular[idx], self.graph_data, self.seq_data[idx]), self.y[idx]

    # Create datasets
    train_dataset = MultiModalDataset(X_tabular_train, graph_data, seq_train, y_train)
    val_dataset = MultiModalDataset(X_tabular_val, graph_data, seq_val, y_val)
    test_dataset = MultiModalDataset(X_tabular_test, graph_data, seq_test, y_test)

    def custom_collate(batch):
        # جداسازی داده‌ها و برچسب‌ها
        inputs = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        
        # جداسازی انواع مختلف داده از inputs
        tabular = torch.stack([item[0] for item in inputs])
        graph = inputs[0][1]  # همه batch‌ها گراف یکسانی دارند
        seq = torch.stack([item[2] for item in inputs])
        
        return (tabular, graph, seq), labels

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=custom_collate
    )

    print(f"\nتعداد دسته‌های آموزشی: {len(train_loader)}")
    print(f"تعداد دسته‌های اعتبارسنجی: {len(val_loader)}")
    print(f"تعداد دسته‌های آزمایشی: {len(test_loader)}")

    # Verify data dimensions
    sample_batch = next(iter(train_loader))
    inputs, labels = sample_batch
    tabular_data, graph_data_batch, seq_data = inputs

    print("\nابعاد داده‌ها در یک batch:")
    print(f"داده‌های جدولی: {tabular_data.shape}")
    print(f"داده‌های گراف: {type(graph_data_batch)}")
    print(f"داده‌های توالی: {seq_data.shape}")
    print(f"برچسب‌ها: {labels.shape}")

except FileNotFoundError as e:
    print(f"خطا در بارگذاری فایل: {e}")
except Exception as e:
    print(f"خطای غیرمنتظره: {e}") 