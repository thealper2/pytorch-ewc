import os

import torch
import typer

from ewc.constants import *
from ewc.data import load_datasets
from ewc.ewc import EWC
from ewc.model import SimpleModel
from ewc.train import evaluate, set_seed, train_epoch
from ewc.utils import plot_results, save_model


def main(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ewc_lambda: float = DEFAULT_EWC_LAMBDA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs_per_task: int = DEFAULT_EPOCHS_PER_TASK,
    seed: int = DEFAULT_SEED,
    fisher_samples: int = DEFAULT_FISHER_SAMPLES,
) -> None:
    """
    Main function to run the EWC experiment.

    Args:
        data_dir: Directory to store datasets
        output_dir: Directory to store output files
        ewc_lambda: Regularization strength parameter for EWC
        learning_rate: Learning rate for optimizer
        epochs_per_task: Number of epochs to train on each task
        seed: Random seed for reproducibility
        fisher_samples: Number of samples for Fisher information calculation
    """
    try:
        # Set random seed for reproducibility
        set_seed(seed)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load datasets
        train_loaders, test_loaders = load_datasets(data_dir)

        # Initialize models
        standard_model = SimpleModel().to(device)
        ewc_model = SimpleModel().to(device)

        # Initialize optimizer for both models
        standard_optimizer = torch.optim.Adam(
            standard_model.parameters(), lr=learning_rate
        )
        ewc_optimizer = torch.optim.Adam(ewc_model.parameters(), lr=learning_rate)

        # Initialize EWC
        ewc = EWC(ewc_model, device)

        # Initialize results dictionaries
        results = {
            "Standard": [],  # Will store (loss, accuracy) tuples for each task
            "EWC": [],  # Will store (loss, accuracy) tuples for each task
        }

        # Define task sequence
        tasks = ["mnist", "fashion"]

        # Train on each task sequentially
        for task_idx, task_name in enumerate(tasks):
            print(
                f"\n{'=' * 50}\nTraining on task {task_idx + 1}: {task_name}\n{'=' * 50}"
            )

            # Get dataloaders for current task
            train_loader = train_loaders[task_name]

            # Train standard model on current task
            print(f"\nTraining standard model on {task_name}...")
            for epoch in range(1, epochs_per_task + 1):
                print(f"Epoch {epoch}/{epochs_per_task}")
                train_loss = train_epoch(
                    standard_model, device, train_loader, standard_optimizer
                )
                print(
                    f"Standard model - Task {task_name} - Epoch {epoch}: Avg loss: {train_loss:.4f}"
                )

            # Train EWC model on current task
            print(f"\nTraining EWC model on {task_name}...")
            for epoch in range(1, epochs_per_task + 1):
                print(f"Epoch {epoch}/{epochs_per_task}")
                train_loss = train_epoch(
                    ewc_model, device, train_loader, ewc_optimizer, ewc, ewc_lambda
                )
                print(
                    f"EWC model - Task {task_name} - Epoch {epoch}: Avg loss: {train_loss:.4f}"
                )

            # Evaluate both models on all tasks seen so far
            print("\nEvaluating models on all tasks seen so far...")
            standard_acc_current = []
            ewc_acc_current = []

            for prev_task_idx, prev_task_name in enumerate(tasks[: task_idx + 1]):
                test_loader = test_loaders[prev_task_name]

                # Evaluate standard model
                std_loss, std_acc = evaluate(
                    standard_model, device, test_loader, f"{prev_task_name} (Standard)"
                )

                # Evaluate EWC model
                ewc_loss, ewc_acc = evaluate(
                    ewc_model, device, test_loader, f"{prev_task_name} (EWC)"
                )

                # Store results for current task
                if prev_task_idx == task_idx:
                    standard_acc_current.append((std_loss, std_acc))
                    ewc_acc_current.append((ewc_loss, ewc_acc))

            # Update results
            results["Standard"].append(standard_acc_current[0])
            results["EWC"].append(ewc_acc_current[0])

            # Save models
            save_model(
                standard_model,
                f"{output_dir}/standard_{MODEL_SAVE_NAME.format(task_name)}",
            )
            save_model(
                ewc_model, f"{output_dir}/ewc_{MODEL_SAVE_NAME.format(task_name)}"
            )

            # Register the current task for EWC
            if task_idx < len(tasks) - 1:  # Don't register after the last task
                print(f"Registering task {task_name} for EWC...")
                ewc.register_task(train_loader, fisher_samples)

        # After training on all tasks, evaluate both models on all tasks
        print("\n\nFinal evaluation on all tasks:")

        # Plot results
        plot_results(
            {
                "Standard Model": [(_, acc) for _, acc in results["Standard"]],
                "EWC Model": [(_, acc) for _, acc in results["EWC"]],
            },
            "Accuracy Comparison: Standard vs EWC",
            f"{output_dir}/{RESULTS_PLOT_NAME}",
        )

        # Evaluate final performance
        final_results = {"Standard": {}, "EWC": {}}

        for task_name in tasks:
            test_loader = test_loaders[task_name]

            # Evaluate standard model
            std_loss, std_acc = evaluate(
                standard_model, device, test_loader, f"Final {task_name} (Standard)"
            )
            final_results["Standard"][task_name] = (std_loss, std_acc)

            # Evaluate EWC model
            ewc_loss, ewc_acc = evaluate(
                ewc_model, device, test_loader, f"Final {task_name} (EWC)"
            )
            final_results["EWC"][task_name] = (ewc_loss, ewc_acc)

        # Calculate forgetting for standard model
        std_forgetting = 0.0
        ewc_forgetting = 0.0

        if len(tasks) > 1:
            # Calculate forgetting for the first task
            first_task = tasks[0]
            std_initial_acc = results["Standard"][0][
                1
            ]  # Accuracy on first task after training on it
            std_final_acc = final_results["Standard"][first_task][
                1
            ]  # Final accuracy on first task
            std_forgetting = std_initial_acc - std_final_acc

            ewc_initial_acc = results["EWC"][0][
                1
            ]  # Accuracy on first task after training on it
            ewc_final_acc = final_results["EWC"][first_task][
                1
            ]  # Final accuracy on first task
            ewc_forgetting = ewc_initial_acc - ewc_final_acc

            print(f"\nForgetting on {first_task}:")
            print(f"Standard Model: {std_forgetting:.2f}%")
            print(f"EWC Model: {ewc_forgetting:.2f}%")
            print(f"Improvement with EWC: {std_forgetting - ewc_forgetting:.2f}%")

        print("\nExperiment completed successfully!")

    except Exception as e:
        print(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    try:
        typer.run(main)
    except Exception as e:
        print(f"Application error: {e}")
        raise
