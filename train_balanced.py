import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from magnet_model import MAGNET

# تعریف دستگاه
device = torch.device('cpu')
print(f"Using device: {device}")

# بارگذاری داده‌های اولیه (بدون SMOTE)
X_tabular_train = torch.load('processed_data/X_tabular_train.pt', weights_only=False)
X_tabular_val = torch.load('processed_data/X_tabular_val.pt', weights_only=False)
graph_data = torch.load('processed_data/graph_data_processed.pt', weights_only=False)
seq_train = torch.load('processed_data/seq_train.pt', weights_only=False)
seq_val = torch.load('processed_data/seq_val.pt', weights_only=False)
y_train = torch.load('processed_data/y_train.pt', weights_only=False)
y_val = torch.load('processed_data/y_val.pt', weights_only=False)

# پاکسازی برچسب‌های عجیب
mask_train = (y_train == 0) | (y_train == 1)
mask_val = (y_val == 0) | (y_val == 1)
X_tabular_train = X_tabular_train[mask_train]
seq_train = seq_train[mask_train]
y_train = y_train[mask_train]
X_tabular_val = X_tabular_val[mask_val]
seq_val = seq_val[mask_val]
y_val = y_val[mask_val]

# بارگذاری بهترین هایپرپارامترها
import json
with open('processed_data/best_config.json', 'r') as f:
    config = json.load(f)

# تعریف تابع collate سفارشی
def custom_collate(batch):
    tabular_data = [item[0][0] for item in batch]
    graph_data = batch[0][0][1]
    seq_data = [item[0][2] for item in batch]
    targets = [item[1] for item in batch]
    return (torch.stack(tabular_data), graph_data, torch.stack(seq_data)), torch.stack(targets)

# تعریف Dataset
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

train_dataset = MultiModalDataset(X_tabular_train, graph_data, seq_train, y_train)
val_dataset = MultiModalDataset(X_tabular_val, graph_data, seq_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate)

# تعریف مدل
model = MAGNET(
    tabular_dim=X_tabular_train.size(1),
    graph_node_dim=graph_data.x.size(1),
    graph_edge_dim=graph_data.edge_attr.size(1),
    seq_vocab_size=215,
    embedding_dim=config['embedding_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    dim_feedforward=config['dim_feedforward'],
    dropout=config['dropout'],
    seq_max_len=seq_train.size(1)
).to(device)

# تعریف تابع ضرر با وزن‌های جدید
class_weights = torch.FloatTensor([1.2, 0.8]).to(device)  # وزن متعادل‌تر
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config['learning_rate'] * 0.01)

# تنظیمات Early Stopping
patience = 5
early_stop_counter = 0
best_val_f1 = 0.0
best_model_state = None

# حلقه آموزش
num_epochs = 50
for epoch in range(num_epochs):
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
    
    # ارزیابی روی مجموعه اعتبارسنجی
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for (tabular, graph, seq), targets in val_loader:
            tabular, graph, seq, targets = tabular.to(device), graph.to(device), seq.to(device), targets.to(device)
            outputs, _, _, _ = model(tabular, graph, seq)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_f1 = f1_score(all_targets, all_preds)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Validation F1 Score: {val_f1:.4f}")
    
    # Early Stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("بهترین مدل به‌روزرسانی شد.")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break
    
    scheduler.step()

# ذخیره بهترین مدل
torch.save(best_model_state, 'processed_data/best_model_balanced.pt')
print("بهترین مدل با وزن‌های متعادل ذخیره شد.")