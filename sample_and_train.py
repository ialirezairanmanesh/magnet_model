import argparse
import random
import numpy as np
import os
from typing import Tuple, List, Dict, Any, Union

# Import your existing modules
# Adjust these imports based on your actual project structure
import read_data  # Import the entire module instead of a specific function
from train_model import train_model

def sample_dataset(data: Union[List, np.ndarray], percentage: float) -> Union[List, np.ndarray]:
    """
    Randomly sample a percentage of the dataset.
    
    Args:
        data: The complete dataset
        percentage: Percentage of data to sample (0-100)
        
    Returns:
        Sampled subset of the dataset
    """
    if percentage <= 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")
    
    if isinstance(data, list):
        sample_size = max(1, int(len(data) * percentage / 100))
        return random.sample(data, sample_size)
    elif isinstance(data, np.ndarray):
        sample_size = max(1, int(len(data) * percentage / 100))
        indices = np.random.choice(len(data), sample_size, replace=False)
        return data[indices]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def sample_and_train(data_path: str, percentage: float, **kwargs) -> Any:
    """
    Load dataset, sample a percentage of it, and train the model.
    
    Args:
        data_path: Path to the dataset
        percentage: Percentage of data to use for training (0-100)
        **kwargs: Additional arguments to pass to the train_model function
        
    Returns:
        Trained model or training results
    """
    # Load the complete dataset
    # Use the appropriate function from the read_data module
    # Depending on what's available in your read_data.py file, use one of these:
    try:
        # Try possible function names
        if hasattr(read_data, 'load_data'):
            dataset = read_data.load_data(data_path)
        elif hasattr(read_data, 'get_data'):
            dataset = read_data.get_data(data_path)
        elif hasattr(read_data, 'read_dataset'):
            dataset = read_data.read_dataset(data_path)
        else:
            # If none of the common names exist, get all functions
            functions = [f for f in dir(read_data) if callable(getattr(read_data, f)) and not f.startswith('_')]
            if functions:
                # Use the first available function as a fallback
                dataset = getattr(read_data, functions[0])(data_path)
            else:
                raise AttributeError("No suitable data loading function found in read_data module")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Available functions in read_data module:")
        for f in dir(read_data):
            if callable(getattr(read_data, f)) and not f.startswith('_'):
                print(f"  - {f}")
        raise
    
    # Sample the dataset
    sampled_dataset = sample_dataset(dataset, percentage)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Sampled dataset size: {len(sampled_dataset)} ({percentage}%)")
    
    # Train the model on the sampled dataset
    result = train_model(sampled_dataset, **kwargs)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Train model on a random sample of the dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--percentage", type=float, default=10.0, 
                        help="Percentage of data to use for training (default: 10.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Add any other arguments that your train_model function requires
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    # parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Extract kwargs for train_model
    kwargs = vars(args)
    del kwargs["data_path"]
    del kwargs["percentage"]
    del kwargs["seed"]
    
    # Sample the dataset and train the model
    result = sample_and_train(args.data_path, args.percentage, **kwargs)
    
    print("Training completed successfully.")
    # You might want to save the result or print a summary here

if __name__ == "__main__":
    main() 