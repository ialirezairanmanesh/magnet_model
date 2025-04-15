import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from magnet_model import MAGNET
from create_dataloaders import MultiModalDataset, custom_collate
from utils import set_seed

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data and return metrics
    """
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch data
            inputs, y = batch
            X_tabular, graph_data, seq_data = inputs
            
            # Move data to device
            X_tabular = X_tabular.to(device)
            graph_data = graph_data.to(device)
            seq_data = seq_data.to(device)
            y = y.to(device)
            
            # Forward pass
            outputs = model(X_tabular, graph_data, seq_data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take the main output if multiple outputs exist
            
            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            
            # Store results
            predictions.append(preds.cpu().numpy())
            targets.append(y.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    probabilities = np.concatenate(probabilities)
    
    # Print shapes for debugging
    print(f"\nShapes before metrics calculation:")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Handle binary classification case
    # If predictions is 2D with 2 columns, use the second column (probability of class 1)
    if predictions.shape[1] == 2:
        # For binary metrics, use the positive class (column 1) predictions
        y_pred = predictions[:, 1]
        y_prob = probabilities[:, 1]
    else:
        y_pred = predictions
        y_prob = probabilities
    
    # Ensure binary format (0 or 1)
    y_pred = y_pred.astype(int)
    targets = targets.astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(targets, y_pred),
        'precision': precision_score(targets, y_pred, zero_division=0),
        'recall': recall_score(targets, y_pred, zero_division=0),
        'f1': f1_score(targets, y_pred, zero_division=0),
        'auc': roc_auc_score(targets, y_prob),
        'confusion_matrix': confusion_matrix(targets, y_pred)
    }
    
    return metrics

def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model configuration
    model_path = "results/magnet_final_20250409_005430/magnet_best_model.pth"
    config_path = "results/magnet_final_20250409_005430/config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load data
    X_tabular_test = torch.load('processed_data/X_tabular_test.pt')
    graph_data = torch.load('processed_data/graph_data_processed.pt', weights_only=False)
    seq_test = torch.load('processed_data/seq_test.pt')
    y_test = torch.load('processed_data/y_test.pt')
    
    # Print data shapes for debugging
    print("\nData shapes:")
    print(f"X_tabular_test shape: {X_tabular_test.shape}")
    print(f"seq_test shape: {seq_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Get dimensions from data
    tabular_dim = X_tabular_test.shape[1]
    graph_node_dim = graph_data.num_node_features
    graph_edge_dim = graph_data.num_edge_features
    seq_vocab_size = 1000  # From saved model
    seq_max_len = 100     # From saved model
    
    # Initialize model with correct parameters
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
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset and loader
    test_dataset = MultiModalDataset(X_tabular_test, graph_data, seq_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save confusion matrix plot
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        "results/test_confusion_matrix.png"
    )
    
    # Save results to JSON
    results_data = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'auc': float(metrics['auc']),
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open("results/test_results.json", 'w') as f:
        json.dump(results_data, f, indent=4)

if __name__ == "__main__":
    main() 