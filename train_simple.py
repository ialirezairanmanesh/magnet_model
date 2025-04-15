#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
نسخه ساده‌تر MAGNET بدون یادگیری خودنظارتی
"""

import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# وارد کردن مدل MAGNET و داده‌ها
from magnet_model import MAGNET, MultiModalDataset, custom_collate_fn
from data_extraction import load_processed_data

# تنظیم دستگاه
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"استفاده از دستگاه: {device}")

# تابع آموزش ساده‌تر بدون SSL
def train_epoch_simple(model, train_loader, criterion, optimizer, device):
    """آموزش مدل برای یک اپوک بدون SSL"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc="آموزش", leave=False) as pbar:
        for batch in pbar:
            (tabular, graph, seq), targets = batch
            tabular = tabular.to(device)
            if hasattr(graph, 'to'):
                graph = graph.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # فقط از logits استفاده می‌کنیم و SSL را نادیده می‌گیریم
            logits, _, _, _ = model(tabular, graph, seq)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * correct / total:.2f}%"
            )
    
    return total_loss / len(train_loader), correct / total

# اجرای با این تابع جدید
def main(args):
    # بارگذاری داده‌ها
    train_data, val_data = load_processed_data()
    
    # ساخت DataLoader
    train_dataset = MultiModalDataset(train_data)
    val_dataset = MultiModalDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=custom_collate_fn
    )

    # ساخت مدل
    model = MAGNET(
        tabular_dim=train_dataset.tabular_dim,
        num_graph_features=train_dataset.num_graph_features,
        seq_dim=train_dataset.seq_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    # تعریف تابع هزینه و بهینه‌ساز
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # حلقه آموزش
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_simple(
            model, train_loader, criterion, optimizer, device
        )
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='آموزش ساده مدل MAGNET')
    parser.add_argument('--batch_size', type=int, default=32, help='اندازه batch')
    parser.add_argument('--epochs', type=int, default=10, help='تعداد epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='نرخ یادگیری')
    parser.add_argument('--hidden_dim', type=int, default=128, help='ابعاد لایه مخفی')
    args = parser.parse_args()
    main(args)