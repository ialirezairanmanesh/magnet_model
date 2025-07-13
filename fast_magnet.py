"""
نسخه سبک‌شده MAGNET برای توسعه سریع
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class FastMAGNET(nn.Module):
    """نسخه سبک‌شده MAGNET برای توسعه سریع - با کاهش چشمگیر تعداد پارامترها"""
    
    def __init__(self, tabular_dim, graph_node_dim, graph_edge_dim, seq_vocab_size, seq_max_len,
                 embedding_dim=32, num_heads=2, num_layers=2, dim_feedforward=64, 
                 dropout=0.2, num_classes=2):
        super(FastMAGNET, self).__init__()
        
        # Tabular Encoder (ساده‌سازی شده)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph Encoder (ساده‌سازی شده - استفاده از GCN به جای TransformerConv)
        self.graph_conv1 = GCNConv(graph_node_dim, embedding_dim)
        self.graph_conv2 = GCNConv(embedding_dim, embedding_dim)
        
        # Sequence Encoder (ساده‌سازی شده)
        self.seq_embedding = nn.Embedding(seq_vocab_size, embedding_dim)
        self.seq_encoder = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        
        # Fusion Layer (ساده‌سازی شده)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tabular, graph, seq):
        # Tabular Path
        tab_emb = self.tabular_encoder(tabular)
        
        # Graph Path
        x, edge_index = graph.x, graph.edge_index
        x = self.graph_conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.graph_conv2(x, edge_index)
        graph_emb = global_mean_pool(x, graph.batch if hasattr(graph, 'batch') else None)
        
        # اگر graph_emb یک تنسور با بعد 2 و اندازه اول 1 باشد، آن را گسترش می‌دهیم
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
            graph_emb = graph_emb.expand(tabular.size(0), -1)
        
        # Sequence Path
        seq_emb = self.seq_embedding(seq)
        _, h_n = self.seq_encoder(seq_emb)
        seq_emb = h_n.squeeze(0)
        
        # Fusion & Classification
        combined = torch.cat([tab_emb, graph_emb, seq_emb], dim=1)
        fused = self.fusion(combined)
        fused = torch.relu(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        # برای سازگاری با کد اصلی، خروجی‌های مشابه برمی‌گردانیم
        dummy_self_supervised = torch.zeros_like(fused)  # خروجی مصنوعی برای SSL
        
        return logits, dummy_self_supervised, None, None  # آخری‌ها برای attention weights 