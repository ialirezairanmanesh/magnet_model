import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from magnet_model import MAGNET
from create_dataloaders import (
    train_loader, val_loader, X_tabular_train, 
    graph_data, seq_train, config, y_train,
    label_map
)

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# تعریف مدل با بهترین پارامترها
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
    seq_max_len=seq_train.size(1),
    num_classes=2
).to(device)

# تنظیم وزن‌های کلاس برای داده‌های نامتوازن
class_counts = torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# تعریف بهینه‌ساز و scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=config['learning_rate'] * 0.01
)

# تنظیمات Early Stopping
patience = 5
early_stop_counter = 0
best_val_f1 = 0.0
best_model_state = None

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, ((tabular, graph, seq), targets) in enumerate(train_loader):
        # انتقال داده‌ها به device
        tabular = tabular.to(device)
        graph = graph.to(device)
        seq = seq.to(device)
        targets = targets.to(device)
        
        # آموزش
        optimizer.zero_grad()
        outputs, _, _, _ = model(tabular, graph, seq)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for (tabular, graph, seq), targets in val_loader:
            tabular = tabular.to(device)
            graph = graph.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            outputs, _, _, _ = model(tabular, graph, seq)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return f1_score(all_targets, all_preds, average='weighted')

# حلقه اصلی آموزش
num_epochs = 50
print("شروع آموزش...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # آموزش یک epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # ارزیابی
    val_f1 = evaluate(model, val_loader, device)
    
    print(f"Train Loss: {train_loss:.4f}, Validation F1 Score: {val_f1:.4f}")
    
    # Early Stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("✓ بهترین مدل به‌روزرسانی شد!")
    else:
        early_stop_counter += 1
        print(f"! {early_stop_counter} epoch بدون بهبود")
        if early_stop_counter >= patience:
            print("\nEarly stopping triggered!")
            break
    
    scheduler.step()

# ذخیره بهترین مدل
torch.save(best_model_state, 'processed_data/best_model.pt')
print("\nآموزش به پایان رسید. بهترین مدل ذخیره شد.")
print(f"بهترین F1 Score در validation: {best_val_f1:.4f}") 