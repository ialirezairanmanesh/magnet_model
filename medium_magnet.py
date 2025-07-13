"""
نسخه متوسط MAGNET: تعادل بین سرعت و دقت
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data

class TabularEncoder(nn.Module):
    """
    انکودر ساده‌تر برای داده‌های جدولی - استفاده از لایه‌های خطی به جای ترانسفورمر
    """
    def __init__(self, input_dim, embedding_dim, dropout=0.2):
        super(TabularEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.layers(x), None  # برای سازگاری با API، None برای attention

class GraphEncoder(nn.Module):
    """
    انکودر گراف با ترکیب GCN و TransformerConv
    """
    def __init__(self, node_dim, edge_dim, embedding_dim, dropout=0.2):
        super(GraphEncoder, self).__init__()
        # لایه اول: GCN ساده‌تر
        self.conv1 = GCNConv(node_dim, embedding_dim * 2)
        # لایه دوم: TransformerConv برای توجه
        self.conv2 = TransformerConv(embedding_dim * 2, embedding_dim, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        # لایه اول
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # لایه دوم
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm(x)
        x = F.relu(x)
        
        # پولینگ گراف
        graph_emb = global_mean_pool(x, graph.batch if hasattr(graph, 'batch') else None)
        
        return graph_emb

class SequenceEncoder(nn.Module):
    """
    انکودر دنباله با ترکیب Embedding و LSTM
    """
    def __init__(self, vocab_size, max_len, embedding_dim, dropout=0.2):
        super(SequenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

class MediumMAGNET(nn.Module):
    """
    نسخه متوسط MAGNET با تعادل بین سرعت و دقت
    """
    def __init__(self, tabular_dim, graph_node_dim, graph_edge_dim, seq_vocab_size, seq_max_len,
                 embedding_dim=64, dropout=0.3, num_classes=2):
        super(MediumMAGNET, self).__init__()
        
        # انکودرهای مدالیته‌ها
        self.tabular_encoder = TabularEncoder(tabular_dim, embedding_dim, dropout)
        self.graph_encoder = GraphEncoder(graph_node_dim, graph_edge_dim, embedding_dim, dropout)
        self.seq_encoder = SequenceEncoder(seq_vocab_size, seq_max_len, embedding_dim, dropout)
        
        # لایه ترکیب
        fusion_input_dim = embedding_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # طبقه‌بندی‌کننده نهایی
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # یادگیری خودنظارت ساده
        self.ssl_head = nn.Linear(embedding_dim, fusion_input_dim)
        
    def forward(self, tabular, graph, seq):
        # پردازش هر مدالیته
        tab_emb, tab_attention = self.tabular_encoder(tabular)
        graph_emb = self.graph_encoder(graph)
        seq_emb = self.seq_encoder(seq)
        
        # اگر graph_emb یک تنسور با بعد 2 و اندازه اول 1 باشد، آن را گسترش می‌دهیم
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
            graph_emb = graph_emb.expand(tabular.size(0), -1)
        
        # ترکیب ویژگی‌ها
        combined = torch.cat([tab_emb, graph_emb, seq_emb], dim=1)
        fused = self.fusion(combined)
        
        # خروجی طبقه‌بندی
        logits = self.classifier(fused)
        
        # خروجی یادگیری خودنظارت
        ssl_output = self.ssl_head(fused)
        
        return logits, ssl_output, tab_attention, None  # برای سازگاری با API 