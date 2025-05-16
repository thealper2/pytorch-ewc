import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from ewc.constants import *


def save_model(model: torch.nn.Module, filename: str) -> None:
    """
    Save model parameters to a file.

    Args:
        model: The neural network model to save
        filename: Path to save the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save model
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def load_model(model: torch.nn.Module, filename: str) -> torch.nn.Module:
    """
    Load model parameters from a file.

    Args:
        model: The neural network model to load parameters into
        filename: Path to the saved model

    Returns:
        Model with loaded parameters
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Model file {filename} not found")
            raise FileNotFoundError(f"Model file {filename} not found")

        # Load model
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def plot_results(
    results: Dict[str, List[Tuple[float, float]]], title: str, filename: str
) -> None:
    """
    Plot training results.

    Args:
        results: Dictionary with results (loss, accuracy) for each model
        title: Title for the plot
        filename: Path to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))

        # Plot accuracy for each model
        for model_name, data in results.items():
            accuracies = [acc for _, acc in data]
            plt.plot(
                range(1, len(accuracies) + 1), accuracies, marker="o", label=model_name
            )

        plt.title(title)
        plt.xlabel("Task Number")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)

        # Save plot
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"Results plot saved to {filename}")
    except Exception as e:
        print(f"Error plotting results: {e}")
        raise
