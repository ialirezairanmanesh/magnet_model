import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from magnet_model import MAGNET  # فرض می‌کنیم مدل توی magnet_model.py هست
import torch.nn.functional as F
# Add these imports for handling PyTorch Geometric data safely
from torch_geometric.data.data import DataEdgeAttr
import torch.serialization
# Add the safe globals for PyTorch Geometric
torch.serialization.add_safe_globals([DataEdgeAttr])
import numpy as np

# مطمئن بشیم مدل روی CPU اجرا می‌شه
device = torch.device('cpu')
print(f"Using device: {device}")

# Custom collate function for handling PyTorch Geometric Data objects
def custom_collate(batch):
    tabular_data = torch.stack([item[0][0] for item in batch])
    graph_data = batch[0][0][1]  # Since graph data is shared across all samples
    seq_data = torch.stack([item[0][2] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return (tabular_data, graph_data, seq_data), targets

# تعریف کلاس Dataset برای داده‌های چندوجهی
class MultiModalDataset(Dataset):
    def __init__(self, X_tabular, graph_data, seq_data, y):
        self.X_tabular = X_tabular
        self.graph_data = graph_data
        self.seq_data = seq_data
        
        # Normalize labels to ensure they are 0 or 1
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        y = np.array(y)
        y = y.flatten()
        y = np.where(y < 0, 0, y)
        y = np.where(y > 1, 1, y)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X_tabular[idx], self.graph_data, self.seq_data[idx]), self.y[idx]

# تابع آموزش برای Optuna (نسخه سبک)
def train_and_evaluate_magnet(trial, X_tabular_train, X_tabular_test, graph_data, seq_train, seq_test, y_train, y_test):
    # تعریف هایپرپارامترها با Optuna (دامنه محدودتر برای سیستم شما)
    config = {
        'embedding_dim': trial.suggest_categorical('embedding_dim', [16, 32, 64]),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True),
        'num_epochs': 5  # تعداد epochها رو کم کردیم
    }

    # مطمئن بشیم embedding_dim مضرب num_heads باشه
    while config['embedding_dim'] % config['num_heads'] != 0:
        config['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])

    # بارگذاری داده‌ها
    train_dataset = MultiModalDataset(X_tabular_train, graph_data, seq_train, y_train)
    test_dataset = MultiModalDataset(X_tabular_test, graph_data, seq_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate)
    
    # تعریف مدل MAGNET
    model = MAGNET(
        tabular_dim=X_tabular_train.size(1),
        graph_node_dim=graph_data.x.size(1),
        graph_edge_dim=graph_data.edge_attr.size(1),
        seq_vocab_size=100,  # تعداد APIها یا توکن‌ها توی واژگان (با داده‌های واقعی تنظیم کن)
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        seq_max_len=seq_train.size(1)
    ).to(device)
    
    # تعریف تابع ضرر و بهینه‌ساز
    # Calculate class weights based on the normalized labels
    y_train_tensor = train_dataset.y
    class_counts = torch.bincount(y_train_tensor)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=config['learning_rate'] * 0.01)
    
    # حلقه آموزش
    best_val_f1 = 0.0
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        
        for (tabular, graph, seq), targets in train_loader:
            tabular, graph, seq, targets = tabular.to(device), graph.to(device), seq.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _, _, _ = model(tabular, graph, seq)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # ارزیابی
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for (tabular, graph, seq), targets in test_loader:
                tabular, graph, seq, targets = tabular.to(device), graph.to(device), seq.to(device), targets.to(device)
                outputs, _, _, _ = model(tabular, graph, seq)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
        scheduler.step()
    
    return best_val_f1

# تابع هدف برای Optuna
def objective(trial):
    # بارگذاری داده‌ها
    X_tabular_train = torch.load('processed_data/X_tabular_train.pt', weights_only=False)
    X_tabular_test = torch.load('processed_data/X_tabular_test.pt', weights_only=False)
    graph_data = torch.load('processed_data/graph_data_processed.pt', weights_only=False)
    seq_train = torch.load('processed_data/seq_train.pt', weights_only=False)
    seq_test = torch.load('processed_data/seq_test.pt', weights_only=False)
    y_train = torch.load('processed_data/y_train.pt', weights_only=False)
    y_test = torch.load('processed_data/y_test.pt', weights_only=False)
    
    # رفع هشدارهای کپی تنسور
    seq_train = seq_train.clone().detach()
    seq_test = seq_test.clone().detach()
    
    # اجرای تابع آموزش و برگرداندن F1 Score
    val_f1 = train_and_evaluate_magnet(trial, X_tabular_train, X_tabular_test, graph_data, seq_train, seq_test, y_train, y_test)
    return val_f1

# اجرای Optuna برای جستجوی بهترین هایپرپارامترها
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# نمایش بهترین هایپرپارامترها
print("بهترین هایپرپارامترها:")
print(study.best_params)
print(f"بهترین F1 Score: {study.best_value:.4f}")

# ذخیره بهترین هایپرپارامترها
import json
with open('processed_data/best_config.json', 'w') as f:
    json.dump(study.best_params, f)

print("بهترین هایپرپارامترها ذخیره شدند.")