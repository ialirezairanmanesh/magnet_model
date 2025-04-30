#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAGNET: تست مدل روی دیتاست DroidRL
این اسکریپت بهترین مدل آموزش دیده را روی دیتاست DroidRL ارزیابی می‌کند
و نتایج را به صورت کامل گزارش می‌دهد.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals
from sklearn.model_selection import train_test_split

# اضافه کردن StandardScaler به لیست globals امن
add_safe_globals([StandardScaler])

class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

def load_droidrl_data():
    """بارگذاری داده‌های DroidRL"""
    print("بارگذاری داده‌های DroidRL...")
    
    # بارگذاری داده‌ها
    data = np.loadtxt('dataset/static_features_latest_new_4gram_latest.csv', delimiter=',')
    X = data[:, :-1]  # همه ستون‌ها به جز آخرین ستون
    y = data[:, -1]   # آخرین ستون برچسب است
    
    print(f"تعداد کل نمونه‌ها: {len(X)}")
    print(f"تعداد ویژگی‌ها: {X.shape[1]}")
    print(f"تعداد نمونه‌های بدافزار: {np.sum(y == 1)}")
    print(f"تعداد نمونه‌های خوش‌خیم: {np.sum(y == 0)}")
    
    # تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, device, num_epochs=50):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor).squeeze().cpu().numpy()
        predictions = (outputs > 0.5).astype(int)
        
    results = {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions)),
        'recall': float(recall_score(y_test, predictions)),
        'f1': float(f1_score(y_test, predictions)),
        'auc': float(roc_auc_score(y_test, outputs))
    }
    
    # رسم و ذخیره ماتریس درهم‌ریختگی
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('ماتریس درهم‌ریختگی')
    plt.ylabel('برچسب واقعی')
    plt.xlabel('برچسب پیش‌بینی شده')
    
    # ایجاد دایرکتوری برای ذخیره نتایج
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/droidrl_test_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # ذخیره نتایج
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nنتایج ارزیابی:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return results

def main():
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_droidrl_data()
    
    # Create and train model
    model = SimpleClassifier(input_size=X_train.shape[1]).to(device)
    train_model(model, X_train, y_train, device)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, device)
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/droidrl_test_{timestamp}'
    torch.save(model.state_dict(), f'{results_dir}/model.pt')
    
    print(f"\nResults and model saved in {results_dir}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main() 