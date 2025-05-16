import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation to prevent catastrophic forgetting.

    EWC adds a penalty to the loss function for changes to parameters that are important
    for the previous task. The importance of each parameter is determined by calculating
    the Fisher information matrix.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize EWC with a model and device.

        Args:
            model: PyTorch model to apply EWC to
            device: Device to perform calculations on (CPU/GPU)
        """
        self.model = model
        self.device = device
        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}  # Mean values of parameters after training on previous task
        self._fisher = {}  # Fisher information matrix diagonal values

    def _update_mean_params(self) -> None:
        """Update the mean parameter values from the current model parameters."""
        for n, p in self.params.items():
            self._means[n] = p.data.clone().detach()

    def _calculate_fisher(self, dataloader: DataLoader, samples: int = 1000) -> None:
        """
        Calculate the Fisher information matrix diagonal values.

        The Fisher information is calculated as the expected squared gradient of the
        log-likelihood with respect to the model parameters.

        Args:
            dataloader: DataLoader with the dataset to calculate Fisher information on
            samples: Number of samples to use for calculating Fisher information
        """
        # Initialize Fisher information for each parameter to zero
        self._fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        # Set model to evaluation mode
        self.model.eval()

        # Process at most 'samples' number of samples
        processed_samples = 0

        for input_data, target in dataloader:
            if processed_samples >= samples:
                break

            batch_size = input_data.size(0)
            if processed_samples + batch_size > samples:
                # Adjust batch size to exactly match the remaining samples needed
                input_data = input_data[: samples - processed_samples]
                target = target[: samples - processed_samples]
                batch_size = samples - processed_samples

            processed_samples += batch_size

            input_data, target = input_data.to(self.device), target.to(self.device)

            # Clear gradients
            self.model.zero_grad()

            # Forward pass
            output = self.model(input_data)

            # For each sample, calculate and accumulate gradients
            for i in range(batch_size):
                # Calculate log probabilities
                log_probs = F.log_softmax(output[i].unsqueeze(0), dim=1)

                # Calculate gradient of the log probability of the target class
                loss = -log_probs[0, target[i]]
                loss.backward(retain_graph=(i < batch_size - 1))

                # Accumulate squared gradients in Fisher information matrix
                for n, p in self.params.items():
                    if p.grad is not None:
                        self._fisher[n] += p.grad.detach() ** 2 / samples

                # Clear gradients for next sample
                self.model.zero_grad()

        print(f"Fisher information matrix calculated using {processed_samples} samples")

    def register_task(self, dataloader: DataLoader, samples: int = 1000) -> None:
        """
        Register a task by calculating Fisher information and storing parameter means.

        This should be called after training on a task before moving to the next task.

        Args:
            dataloader: DataLoader containing the dataset for the current task
            samples: Number of samples to use for calculating Fisher information
        """
        try:
            print("Registering task for EWC...")
            self._calculate_fisher(dataloader, samples)
            self._update_mean_params()
            print("Task registered successfully")
        except Exception as e:
            print(f"Error registering task: {e}")
            raise

    def ewc_loss(self, lamda: float = 40.0) -> torch.Tensor:
        """
        Calculate the EWC loss (regularization term).

        Args:
            lamda: Regularization strength parameter

        Returns:
            EWC loss value as a tensor
        """
        if not self._fisher or not self._means:
            return torch.tensor(0.0).to(self.device)

        loss = torch.tensor(0.0).to(self.device)

        # For each parameter, calculate the squared difference from the mean,
        # weighted by the Fisher information
        for n, p in self.params.items():
            # Calculate the EWC penalty for this parameter
            _loss = self._fisher[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return 0.5 * lamda * loss
