"""
MAGNET: Multi-Modal Attention-Graph-Embedding Transformer for Android Malware Detection
Author: [Your Name]
Date: March 09, 2025 (Updated: [Current Date])
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# torch_geometric imports for graph processing
from torch_geometric.data import Data # Used for type hinting and potentially later extensions
from torch_geometric.nn import TransformerConv, global_mean_pool
import torch.nn.functional as F2
from sklearn.model_selection import train_test_split
from data_extraction import load_processed_data

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset wrappers
class MultiModalDataset(Dataset):
    """
    Dataset wrapper for tabular, graph, and sequence data.
    IMPORTANT: Assumes a *single, shared* graph_data object for all samples.
    If each sample has its own graph, this class and the DataLoader need modification
    (e.g., using torch_geometric.loader.DataLoader).
    """
    def __init__(self, X_tabular, graph_data: Data, seq_data, y):
        if not isinstance(X_tabular, torch.Tensor):
            self.X_tabular = torch.FloatTensor(X_tabular)
        else:
            self.X_tabular = X_tabular.float()

        self.graph_data = graph_data # Store the single PyG graph object

        if not isinstance(seq_data, torch.Tensor):
            # Sequence data should typically be integers (token IDs)
            self.seq_data = torch.LongTensor(seq_data)
        else:
            # Ensure it's Long type for embedding layers
            self.seq_data = seq_data.long()

        if not isinstance(y, torch.Tensor):
             # Ensure y is a LongTensor for CrossEntropyLoss
            self.y = torch.LongTensor(y.values if isinstance(y, pd.Series) else y)
        else:
            self.y = y.long() # Ensure Long type

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Note: graph_data is the same for all items due to the single graph assumption
        return (self.X_tabular[idx], self.graph_data, self.seq_data[idx]), self.y[idx]

# Dynamic Attention Mechanism
class DynamicAttention(nn.Module):
    """Dynamic attention with learnable weights for feature importance"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        # Learnable scalar importance weight for the attention scores
        self.importance_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_length = x.shape[1] # This is the sequence length or number of features/modalities

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Reshape for multi-head attention
        queries = queries.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores (energy)
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if mask is not None:
            # Apply mask (e.g., for padding)
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # Apply softmax with learnable importance weight
        # Ensure broadcasting works correctly: energy shape (batch, heads, seq, seq), importance_weight shape (1)
        attention = torch.softmax(energy * self.importance_weight, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention, values)

        # Reshape back to (batch_size, seq_length, d_model)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)
        output = self.out(output)

        return output, attention # Return attention weights for interpretability

# Enhanced TabTransformer for Tabular Data
class EnhancedTabTransformer(nn.Module):
    """Transformer for tabular data, processing features as tokens."""
    def __init__(self, input_dim, embedding_dim=64, num_heads=8, num_layers=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        # Embed each feature value independently
        # We use a simple linear layer assuming input features are continuous.
        # For categorical features, nn.Embedding would be more appropriate, requiring modifications.
        self.embedding = nn.Sequential(
            nn.Linear(1, embedding_dim), # Embed each feature value
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        # Positional embedding for features (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embedding_dim) * (embedding_dim ** -0.5))
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': DynamicAttention(embedding_dim, num_heads),
                'norm1': nn.LayerNorm(embedding_dim),
                'ff': nn.Sequential(
                    nn.Linear(embedding_dim, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, embedding_dim)
                ),
                'norm2': nn.LayerNorm(embedding_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: (batch_size, num_features)
        batch_size, num_features = x.shape
        # Reshape to (batch_size, num_features, 1) to apply embedding to each feature value
        x = x.unsqueeze(-1)

        # Apply embedding to each feature independently
        embeddings = [self.embedding(x[:, i, :]) for i in range(num_features)]
        x = torch.stack(embeddings, dim=1) # shape: (batch_size, num_features, embedding_dim)

        # Add positional embedding
        x = x + self.pos_embedding # Broadcasting applies the same pos_embedding across the batch
        x = self.input_dropout(x)

        attention_weights = []
        for layer in self.layers:
            # Self-attention over features
            attn_output, attn_weights = layer['attention'](x)
            # Residual connection and normalization
            x = layer['norm1'](x + layer['dropout'](attn_output))
            # Feed-forward network
            ff_output = layer['ff'](x)
            # Residual connection and normalization
            x = layer['norm2'](x + layer['dropout'](ff_output))
            attention_weights.append(attn_weights) # Collect attention weights from each layer

        # Return embeddings per feature and attention weights
        # Shape: (batch_size, num_features, embedding_dim), List of attention tensors
        return x, attention_weights

# Graph Transformer
class GraphTransformer(nn.Module):
    """Transformer for graph data using PyTorch Geometric's TransformerConv."""
    def __init__(self, node_dim, edge_dim, embedding_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        # Linear layers to project node and edge features to the embedding dimension
        self.node_embedding = nn.Linear(node_dim, embedding_dim)
        # Edge features need embedding if edge_dim is provided and > 0
        self.edge_embedding = nn.Linear(edge_dim, embedding_dim) if edge_dim > 0 else None

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # TransformerConv handles graph structure with attention
            self.layers.append(
                TransformerConv(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim // num_heads, # Output channels per head
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=embedding_dim if self.edge_embedding else None # Pass edge embedding dim if used
                )
            )

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Global pooling to get a single graph-level embedding
        self.global_pool = global_mean_pool
        # Final projection for the graph embedding
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, data: Data):
        # data is expected to be a single torch_geometric.data.Data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embed node features
        x = self.node_embedding(x)

        # Embed edge features if they exist
        if self.edge_embedding and edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
        else:
            edge_attr = None # Pass None to TransformerConv if no edge features or embedding

        # Apply TransformerConv layers
        for layer in self.layers:
            # Note: TransformerConv internally handles concatenation or averaging of heads
            x_new = layer(x, edge_index, edge_attr)
            x = x + self.dropout(F2.relu(x_new)) # Residual connection with ReLU

        x = self.norm(x) # Normalize node embeddings

        # Global pooling: Aggregate node embeddings into a single graph embedding
        # Create a batch vector of zeros if data represents a single graph
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.global_pool(x, batch) # Shape: (1, embedding_dim)

        # Final projection
        pooled = self.output_projection(pooled) # Shape: (1, embedding_dim)

        return pooled

# Sequence Transformer
class SequenceTransformer(nn.Module):
    """Transformer for sequence data (e.g., API call sequences)."""
    def __init__(self, vocab_size, embedding_dim=64, num_heads=8, num_layers=4, dim_feedforward=256, dropout=0.2, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # Use padding_idx=0 if 0 is used for padding
        # Fixed sinusoidal positional encoding
        self.pos_encoding = nn.Parameter(self._generate_pos_encoding(max_len, embedding_dim), requires_grad=False)
        self.input_dropout = nn.Dropout(dropout)
        self.max_len = max_len

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': DynamicAttention(embedding_dim, num_heads), # Using the same dynamic attention
                'norm1': nn.LayerNorm(embedding_dim),
                'ff': nn.Sequential(
                    nn.Linear(embedding_dim, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, embedding_dim)
                ),
                'norm2': nn.LayerNorm(embedding_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

    def _generate_pos_encoding(self, max_len, d_model):
        """Generates sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # Shape: (1, max_len, d_model)

    def forward(self, seq):
        # seq shape: (batch_size, seq_length)
        batch_size, seq_len = seq.shape

        # Ensure input is LongTensor for embedding lookup
        if seq.dtype != torch.long:
            seq = seq.long()

        # Truncate or pad sequence if needed (though DataLoader usually handles padding)
        if seq_len > self.max_len:
            seq = seq[:, :self.max_len]
            seq_len = self.max_len
        elif seq_len < self.max_len:
             # Assuming padding value is 0, consistent with padding_idx=0 in nn.Embedding
             padding = torch.zeros((batch_size, self.max_len - seq_len), dtype=torch.long, device=seq.device)
             seq = torch.cat([seq, padding], dim=1)
             seq_len = self.max_len


        # Get embeddings and add positional encoding
        x = self.embedding(seq) # shape: (batch_size, seq_len, embedding_dim)
        # Add positional encoding (slice to match sequence length)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.input_dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            attn_output, _ = layer['attention'](x) # Self-attention over sequence tokens
            x = layer['norm1'](x + layer['dropout'](attn_output)) # Residual + Norm
            ff_output = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_output)) # Residual + Norm

        # Pooling: Average embeddings across the sequence length dimension
        x = x.mean(dim=1) # shape: (batch_size, embedding_dim)

        return x

# Modality Fusion Layer
class ModalityFusion(nn.Module):
    """Fuses embeddings from different modalities using attention."""
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1, num_classes=2): # Added num_classes
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Attention mechanism to weigh modalities relative to each other
        self.cross_modality_attention = DynamicAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Feed-forward layer after attention
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), # Expansion layer
            nn.ReLU(),
            nn.Dropout(dropout), # Added dropout
            nn.Linear(4 * embedding_dim, embedding_dim) # Projection back
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional: Projection head for self-supervised learning task
        # This might predict properties derived from the combined input, or reconstruction.
        # The exact nature of the SSL task needs careful design based on the data.
        self.ssl_projection = nn.Linear(embedding_dim, embedding_dim) # Example projection

        # Final classifier takes the fused representation
        self.classifier = nn.Linear(embedding_dim, self.num_classes) # Use num_classes

        # New SSL head
        self.ssl_head = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)  # تطبیق با ابعاد ورودی
        )

    def forward(self, tab_emb, graph_emb, seq_emb):
        # tab_emb shape: (batch_size, embedding_dim) - After pooling in MAGNET
        # graph_emb shape: (1, embedding_dim) - From GraphTransformer (single graph)
        # seq_emb shape: (batch_size, embedding_dim) - After pooling in SequenceTransformer
        batch_size = tab_emb.shape[0]

        # Expand the single graph embedding to match the batch size
        # This is crucial due to the single-graph assumption
        if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
            graph_emb_expanded = graph_emb.expand(batch_size, -1) # Shape: (batch_size, embedding_dim)
        else:
            # Handle cases where graph_emb might already be batched (e.g., if model structure changes later)
            # Or raise an error if the shape is unexpected under the current single-graph assumption.
             if graph_emb.shape[0] != batch_size or graph_emb.dim() != 2:
                 raise ValueError(f"Unexpected graph_emb shape: {graph_emb.shape}. Expected ({batch_size} or 1, {self.embedding_dim})")
             graph_emb_expanded = graph_emb


        # Stack modality embeddings along a new dimension (dim=1)
        # Shape: (batch_size, num_modalities=3, embedding_dim)
        combined = torch.stack([tab_emb, graph_emb_expanded, seq_emb], dim=1)

        # Apply cross-modality attention
        # The attention mechanism learns to weigh the importance of each modality relative to others.
        attended, fusion_attention = self.cross_modality_attention(combined)
        # Residual connection and normalization
        combined = self.norm1(combined + self.dropout(attended))

        # Apply feed-forward network to each modality's representation independently
        ff_output = self.feed_forward(combined)
        # Residual connection and normalization
        combined = self.norm2(combined + self.dropout(ff_output))

        # Pooling across modalities: Average the representations after fusion
        # This creates a single representation vector for the entire input sample.
        fused = combined.mean(dim=1) # Shape: (batch_size, embedding_dim)

        # Optional: Generate output for self-supervised task
        ssl_output = self.ssl_projection(fused) # Example: Project the fused embedding

        # Final classification layer
        logits = self.classifier(fused) # Shape: (batch_size, num_classes)

        # Return classification logits, SSL output, and attention weights
        return logits, ssl_output, fusion_attention

# Main MAGNET Model
class MAGNET(nn.Module):
    """
    MAGNET: Multi-Modal Attention-Graph-Embedding Transformer.
    Combines tabular, graph (single shared graph assumed), and sequence transformers.
    """
    def __init__(
        self,
        tabular_dim: int,
        graph_node_dim: int,
        graph_edge_dim: int,
        seq_vocab_size: int,
        seq_max_len: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int, # Number of layers for *each* transformer
        dim_feedforward: int,
        dropout: float,
        num_classes: int = 2 # Default to binary classification
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Initialize transformers for each modality
        self.tab_transformer = EnhancedTabTransformer(
            input_dim=tabular_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.graph_transformer = GraphTransformer(
            node_dim=graph_node_dim,
            edge_dim=graph_edge_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.seq_transformer = SequenceTransformer(
            vocab_size=seq_vocab_size,
            max_len=seq_max_len,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Initialize the fusion layer, passing num_classes
        self.fusion = ModalityFusion(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_classes=num_classes # Pass num_classes here
        )

        # Note: The final classifier is now *inside* the ModalityFusion layer.
        # No separate classifier needed here.

    def forward(self, tabular: torch.Tensor, graph: Data, seq: torch.Tensor):
        # Process each modality
        # Tabular: Get feature embeddings and pool them
        tab_emb_features, tab_attention = self.tab_transformer(tabular)
        # Pool across the feature dimension for tabular data
        tab_emb = tab_emb_features.mean(dim=1) # Shape: (batch_size, embedding_dim)

        # Graph: Get graph-level embedding (assumes single graph)
        # Ensure graph data is on the correct device (if not already handled)
        if graph.x.device != self.fusion.classifier.weight.device: # Check device using a model parameter
             graph = graph.to(self.fusion.classifier.weight.device)
        graph_emb = self.graph_transformer(graph) # Shape: (1, embedding_dim)

        # Sequence: Get sequence-level embedding
        seq_emb = self.seq_transformer(seq) # Shape: (batch_size, embedding_dim)

        # Fuse the embeddings from the three modalities
        # The fusion layer handles classification internally now
        logits, self_supervised_output, fusion_attention = self.fusion(tab_emb, graph_emb, seq_emb)

        # Return final classification logits, SSL output, and attention weights
        return logits, self_supervised_output, tab_attention, fusion_attention

# این تابع را قبل از تابع train_and_evaluate_magnet اضافه کنید
def custom_collate_fn(batch):
    """
    تابع سفارشی برای ترکیب داده‌ها در یک بچ
    این تابع به درستی با ترکیبی از داده‌های معمولی و PyTorch Geometric کار می‌کند
    """
    tabular_batch = []
    seq_batch = []
    target_batch = []
    
    # همه نمونه‌ها از گراف یکسان استفاده می‌کنند، پس فقط اولی را می‌گیریم
    graph_data = batch[0][0][1]  # گراف از اولین نمونه
    
    for (tabular, graph, seq), target in batch:
        tabular_batch.append(tabular)
        seq_batch.append(seq)
        target_batch.append(target)
    
    # تبدیل به تنسور
    tabular_batch = torch.stack(tabular_batch)
    seq_batch = torch.stack(seq_batch)
    target_batch = torch.tensor(target_batch)
    
    return (tabular_batch, graph_data, seq_batch), target_batch

# --- Training and Evaluation Function ---
def train_and_evaluate_magnet(config, sample_percentage=100, progress_callback=None):
    """
    Train and evaluate the MAGNET model.

    Args:
        config (dict): Dictionary containing model hyperparameters and data parameters.
                       Expected keys: embedding_dim, num_heads, num_layers, dim_feedforward,
                                      dropout, batch_size, num_epochs, learning_rate,
                                      weight_decay, seq_max_len, num_classes (optional, default 2),
                                      seq_vocab_size (required if not using placeholder)
        sample_percentage (int): Percentage of data to use (1-100). Default is 100.
        progress_callback (callable): Callback function for progress updates.
    """
    print(f"\n--- Starting Training & Evaluation (Sample: {sample_percentage}%) ---")
    # --- 1. Load and Prepare Data ---
    # Replace placeholder with your actual data loading
    X_tabular_train, X_tabular_test, graph_data_single, seq_data_train, seq_data_test, y_train_raw, y_test_raw = \
        load_processed_data() # Use placeholder or your function

    # Check and extract dimensions needed for model initialization
    if X_tabular_train is None:
        raise ValueError("Data loading failed. Check load_processed_data.")

    tabular_dim = X_tabular_train.shape[1]
    graph_node_dim = graph_data_single.x.shape[1]
    # Handle case where edge_attr might be None or empty
    graph_edge_dim = graph_data_single.edge_attr.shape[1] if graph_data_single.edge_attr is not None and graph_data_single.edge_attr.numel() > 0 else 0
    # seq_vocab_size needs to be known. Get from config or infer from data if possible.
    # IMPORTANT: Provide seq_vocab_size in the config!
    if 'seq_vocab_size' not in config:
         # Try to infer from placeholder data, otherwise raise error
         if 'vocab_size' in locals(): # Check if placeholder defined it
              config['seq_vocab_size'] = vocab_size
              print(f"Inferred seq_vocab_size from placeholder: {config['seq_vocab_size']}")
         else:
              raise ValueError("Configuration must include 'seq_vocab_size'.")

    seq_vocab_size = config['seq_vocab_size']
    seq_max_len = config['seq_max_len']
    num_classes = config.get('num_classes', 2) # Default to 2 if not in config

    print("\nData Dimensions:")
    print(f"  Tabular Features: {tabular_dim}")
    print(f"  Graph Node Features: {graph_node_dim}")
    print(f"  Graph Edge Features: {graph_edge_dim}")
    print(f"  Sequence Vocab Size: {seq_vocab_size}")
    print(f"  Sequence Max Length: {seq_max_len}")
    print(f"  Number of Classes: {num_classes}")


    # --- 2. Data Sampling (Optional) ---
    if sample_percentage < 100:
        print(f"\nSampling {sample_percentage}% of the data...")
        train_indices = np.arange(len(y_train_raw))
        test_indices = np.arange(len(y_test_raw))

        # Use train_test_split for stratified sampling to get indices
        _, train_indices = train_test_split(
            train_indices, train_size=sample_percentage/100.0,
            stratify=y_train_raw, random_state=42
        )
        _, test_indices = train_test_split(
            test_indices, test_size=sample_percentage/100.0, # Apply percentage to test set too
            stratify=y_test_raw, random_state=42
        )

        # Apply indices
        X_tabular_train = X_tabular_train[train_indices]
        seq_data_train = seq_data_train[train_indices]
        y_train_raw = y_train_raw.iloc[train_indices] if isinstance(y_train_raw, pd.Series) else y_train_raw[train_indices]

        X_tabular_test = X_tabular_test[test_indices]
        seq_data_test = seq_data_test[test_indices]
        y_test_raw = y_test_raw.iloc[test_indices] if isinstance(y_test_raw, pd.Series) else y_test_raw[test_indices]

        print(f"Sampled {len(y_train_raw)} training and {len(y_test_raw)} test instances.")

    # --- 3. Preprocessing ---
    # Scale tabular data
    scaler = StandardScaler()
    X_tabular_train_scaled = scaler.fit_transform(X_tabular_train)
    X_tabular_test_scaled = scaler.transform(X_tabular_test)

    # Process labels (ensure correct type and range)
    y_train = y_train_raw.values if isinstance(y_train_raw, pd.Series) else np.array(y_train_raw)
    y_test = y_test_raw.values if isinstance(y_test_raw, pd.Series) else np.array(y_test_raw)

    # Ensure labels are integers starting from 0
    y_train = y_train.astype(np.int64).flatten()
    y_test = y_test.astype(np.int64).flatten()

    # Basic validation for labels (e.g., checking range if num_classes is known)
    if np.any(y_train < 0) or np.any(y_train >= num_classes):
        print(f"\nWarning: Training labels outside expected range [0, {num_classes-1}]. Clamping or check data.")
        # Handle appropriately - e.g., clamp or raise error
        y_train = np.clip(y_train, 0, num_classes - 1)

    # Calculate class weights for handling imbalance
    class_counts = np.bincount(y_train, minlength=num_classes)
    print(f"\nClass distribution (Train): {dict(enumerate(class_counts))}")

    # Avoid division by zero if a class has no samples (add small epsilon or handle differently)
    if np.any(class_counts == 0):
        print("\nWarning: One or more classes have zero samples in the training set!")
        # Use equal weights or another strategy if this happens
        # Adding epsilon to avoid division by zero, but equal weights might be safer
        weights = 1.0 / (class_counts + 1e-6)
    else:
        weights = 1.0 / class_counts

    class_weights = torch.FloatTensor(weights / weights.sum() * num_classes).to(device) # Normalize weights
    print(f"Using class weights: {class_weights.cpu().numpy()}")

    # Move the single graph data to the GPU *once*
    graph_data_single = graph_data_single.to(device)
    print(f"Moved single graph data to {graph_data_single.x.device}")


    # --- 4. Create Datasets and DataLoaders ---
    train_dataset = MultiModalDataset(X_tabular_train_scaled, graph_data_single, seq_data_train, y_train)
    test_dataset = MultiModalDataset(X_tabular_test_scaled, graph_data_single, seq_data_test, y_test)

    # استفاده از تابع collate_fn سفارشی
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    # --- 5. Initialize Model ---
    model = MAGNET(
        tabular_dim=tabular_dim,
        graph_node_dim=graph_node_dim,
        graph_edge_dim=graph_edge_dim,
        seq_vocab_size=seq_vocab_size,
        seq_max_len=seq_max_len,
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_classes=num_classes
    ).to(device)

    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- 6. Define Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    self_supervised_criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # بهبود scheduler برای کاهش نرخ یادگیری در صورت عدم بهبود
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # پارامترهای توقف زودهنگام
    early_stop_patience = config.get('early_stop_patience', 10)
    no_improve_epochs = 0
    
    # اضافه کردن این بخش بعد از مقداردهی اولیه پارامترها
    fast_mode = config.get('fast_mode', False)
    if fast_mode:
        print("⚡ حالت سریع فعال است - برخی محاسبات پیچیده حذف می‌شوند ⚡")
        # در حالت سریع، برخی محاسبات پیچیده را رد می‌کنیم
        alpha = 0.0  # غیرفعال کردن SSL
    
    # در بخش آموزش، این تغییرات را اعمال کنید
    skip_validation = config.get('skip_validation', False)
    validation_frequency = config.get('validation_frequency', 1)
    
    # گام‌های تجمیع گرادیان
    grad_accumulation_steps = config.get('grad_accumulation_steps', 1)

    # --- 7. Training Loop ---
    print(f"\n--- Starting Training for {config['num_epochs']} epochs ---")
    best_val_f1 = -1.0 # Track best F1 score for model saving
    best_model_state = None

    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0.0
        total_ssl_loss = 0.0
        optimizer.zero_grad()  # صفر کردن گرادیان‌ها در ابتدای هر epoch

        for i, ((tabular, graph, seq), targets) in enumerate(train_loader):
            # Move batch data to device
            tabular, seq, targets = tabular.to(device), seq.to(device), targets.to(device)
            # Graph is already on device

            # Forward pass
            logits, self_supervised_output, _, _ = model(tabular, graph, seq) # graph is the single shared graph

            # Calculate classification loss
            classification_loss = criterion(logits, targets)

            # Calculate self-supervised loss
            # --- WARNING: SSL Target Calculation ---
            # This target assumes reconstruction of averaged *input* embeddings.
            # It heavily relies on the single-graph structure and might not be optimal.
            # Consider revising the SSL task (e.g., contrastive loss between modalities).
            with torch.no_grad(): # Target should not require gradients
                # Ensure graph_emb_expanded is created correctly (use the logic from ModalityFusion)
                 graph_emb_single = model.graph_transformer(graph) # Get the single graph embedding
                 if graph_emb_single.dim() == 2 and graph_emb_single.size(0) == 1:
                       graph_emb_expanded = graph_emb_single.expand(tabular.size(0), -1)
                 else: # Should not happen with current structure but good practice
                       graph_emb_expanded = graph_emb_single # Assuming already batched somehow

                 # Use pooled embeddings similar to what fusion layer receives
                 tab_emb_pooled = model.tab_transformer(tabular)[0].mean(dim=1)
                 seq_emb_pooled = model.seq_transformer(seq)

                 # Concatenate *pooled* embeddings as the target for the fused representation's SSL projection
                 # Target dimension should match ssl_output dimension (embedding_dim)
                 # A simple target: average of the input modality embeddings
                 ssl_target = (tab_emb_pooled + graph_emb_expanded + seq_emb_pooled) / 3.0
                 # Ensure target shape matches ssl_output shape
                 ssl_target = ssl_target.detach() # Detach from computation graph

            # Check shapes before calculating SSL loss
            if self_supervised_output.shape != ssl_target.shape:
                 print(f"Warning: SSL shape mismatch. Output: {self_supervised_output.shape}, Target: {ssl_target.shape}")
                 # Skip SSL loss if shapes mismatch to avoid crashing
                 self_supervised_loss = torch.tensor(0.0, device=device)
            else:
                 self_supervised_loss = self_supervised_criterion(self_supervised_output, ssl_target)
            # --- End SSL Target Calculation ---


            # Combined loss (adjust the weight alpha)
            loss = classification_loss + alpha * self_supervised_loss
            
            # تقسیم loss برای تجمیع گرادیان
            loss = loss / grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # بروزرسانی وزن‌ها فقط در گام‌های مشخص
            if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += classification_loss.item()
            total_ssl_loss += self_supervised_loss.item() if isinstance(self_supervised_loss, torch.Tensor) else self_supervised_loss


        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for (tabular, graph, seq), targets in test_loader:
                tabular, seq = tabular.to(device), seq.to(device)
                targets = targets.to(device)
                # Graph is already on device

                logits, _, _, _ = model(tabular, graph, seq)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()

                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(test_loader)
        avg_ssl_loss = total_ssl_loss / len(train_loader)

        # Calculate validation metrics
        val_accuracy = accuracy_score(all_targets, all_preds)
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)

        # استفاده از scheduler با معیار F1
        scheduler.step(val_f1)
        
        # ارزیابی مدل
        if not skip_validation or (epoch % validation_frequency == 0):
            metrics = {
                'Accuracy': val_accuracy,
                'Precision': val_precision,
                'Recall': val_recall,
                'F1 Score': val_f1
            }
            
            # به‌روزرسانی نوار پیشرفت
            if progress_callback:
                progress_callback(epoch, avg_val_loss, metrics)
            
            # Save the best model based on validation F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                no_improve_epochs = 0
                print(f"*** New best model saved (Epoch {epoch+1}, Val F1: {best_val_f1:.4f}) ***")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # در اپوک‌هایی که validation نداریم، فقط loss آموزش را نمایش می‌دهیم
            print(f"Epoch [{epoch+1}/{config['num_epochs']}] | Train Loss: {avg_train_loss:.4f}")
    
    # --- 8. Final Evaluation ---
    if best_model_state:
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(best_model_state)
    else:
        print("\nWarning: No best model saved. Evaluating the last epoch model.")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (tabular, graph, seq), targets in test_loader:
            # اصلاح شده - فقط tabular و seq به device منتقل می‌شوند
            tabular = tabular.to(device)
            seq = seq.to(device)
            targets = targets.to(device)
            # graph قبلاً به device منتقل شده است

            logits, _, tab_attention, fusion_attention = model(tabular, graph, seq)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate final metrics
    final_accuracy = accuracy_score(all_targets, all_preds)
    final_precision = precision_score(all_targets, all_preds, zero_division=0)
    final_recall = recall_score(all_targets, all_preds, zero_division=0)
    final_f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    results = {
        'Accuracy': final_accuracy,
        'Precision': final_precision,
        'Recall': final_recall,
        'F1 Score': final_f1,
        'Best Val F1': best_val_f1 # Include the best validation F1 achieved during training
    }

    print("\n--- Final Evaluation Results (Best Model) ---")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # --- 9. Save Model and Scaler ---
    save_path = 'processed_data/magnet_model_final.pth'
    print(f"\nSaving final model, scaler, and config to {save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # Optional but useful
        'scaler_state': scaler.get_params(), # Save scaler parameters
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'config': config, # Save the configuration used
        'results': results # Save the final results
    }, save_path)

    # Return necessary objects for potential further use
    return model, scaler, results # Removed attention_weights return for now

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define configuration for the experiment
    config = {
        # Model Hyperparameters
        'embedding_dim': 64,
        'num_heads': 8,
        'num_layers': 4,      # Layers for Tab/Seq/Graph Transformers
        'dim_feedforward': 256,
        'dropout': 0.2,

        # Training Hyperparameters
        'batch_size': 64,    # Reduced batch size from original, adjust based on GPU memory
        'num_epochs': 50,     # Increase epochs for potentially better convergence
        'learning_rate': 0.0005, # Slightly lower LR
        'weight_decay': 0.01,

        # Data Parameters (Crucial - Must be set correctly for your data!)
        'seq_max_len': 100,    # Adjust based on your sequence data padding/truncation
        'seq_vocab_size': 1000, # *** MUST BE SET based on your sequence data vocabulary ***
                              # This value is critical for the SequenceTransformer's embedding layer.
                              # If using the placeholder, it's set to 1000. Replace for real data.
        'num_classes': 2,      # Set to the number of classes in your dataset (e.g., 2 for malware/benign)
    }

    # Set the percentage of data to use for quick tests (e.g., 10 for 10%)
    # Set to 100 to use the full dataset
    sample_percentage = 10 # Use 10% of data for faster experimentation

    print(f"\n{'='*70}")
    print(f"Initiating MAGNET Model Training")
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"Using {sample_percentage}% of the data.")
    print('='*70)

    # Run the training and evaluation
    # Note: The function now uses the placeholder data loader.
    # Replace load_processed_data call inside the function with your actual loader.
    model, scaler, results = train_and_evaluate_magnet(config, sample_percentage)

    print("\n--- Training and Evaluation Complete ---")