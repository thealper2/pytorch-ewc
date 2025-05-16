from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ewc.ewc import EWC
from ewc.model import SimpleModel


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed value
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: SimpleModel,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ewc: Optional[EWC] = None,
    ewc_lambda: float = 40.0,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model to train
        device: The device to use for training (CPU/GPU)
        train_loader: DataLoader with training data
        optimizer: Optimizer for updating model parameters
        ewc: EWC instance for calculating regularization loss (optional)
        ewc_lambda: Regularization strength parameter for EWC

    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate cross-entropy loss
        loss = F.cross_entropy(output, target)

        # Add EWC penalty if EWC is used
        if ewc is not None:
            ewc_penalty = ewc.ewc_loss(ewc_lambda)
            loss += ewc_penalty

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Log progress
        if batch_idx % 100 == 0:
            print(
                f"Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}"
            )

    # Calculate average loss
    avg_loss = train_loss / len(train_loader)
    return avg_loss


def evaluate(
    model: SimpleModel, device: torch.device, test_loader: DataLoader, dataset_name: str
) -> Tuple[float, float]:
    """
    Evaluate the model on a test dataset.

    Args:
        model: The neural network model to evaluate
        device: The device to use for evaluation (CPU/GPU)
        test_loader: DataLoader with test data
        dataset_name: Name of the dataset for logging

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Calculate loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)

            # Calculate accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    # Calculate average loss and accuracy
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total

    print(
        f"Test {dataset_name}: Average loss: {avg_loss:.4f}, "
        f"Accuracy: {correct}/{total} ({accuracy:.2f}%)"
    )

    return avg_loss, accuracy
