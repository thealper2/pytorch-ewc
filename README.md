# Elastic Weight Consolidation (EWC) Implementation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

This project implements Elastic Weight Consolidation (EWC), a technique to mitigate catastrophic forgetting in neural networks when learning sequential tasks. The implementation is tested on MNIST and Fashion-MNIST datasets.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

This project demonstrates:
- Implementation of EWC algorithm
- Comparison between standard training and EWC
- Evaluation on MNIST and Fashion-MNIST datasets

## Features

- Modular PyTorch implementation
- Configurable hyperparameters
- Automatic dataset downloading
- Model saving and loading
- Training progress visualization
- Comprehensive evaluation metrics

## Installation

1. Clone the repository:

```bash
git clone https://github.com/thealper2/pytorch-ewc.git
cd pytorch-ewc
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/active # On Window use `venv\Scripts\active`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the experiment

```bash
python3 main.py --ewc_lambda 50.0 --epochs_per_task 10 --output_dir ./my_results
```

### Command Line Options

```bash
 Args:     
    data_dir: Directory to store datasets     
    output_dir: Directory to store output files
    ewc_lambda: Regularization strength parameter for EWC     
    learning_rate: Learning rate for optimizer
    epochs_per_task: Number of epochs to train on each task     
    seed: Random seed for reproducibility
    fisher_samples: Number of samples for Fisher information calculation

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --data-dir               TEXT     [default: ./data]                                                      │
│ --output-dir             TEXT     [default: ./output]                                                    │
│ --ewc-lambda             FLOAT    [default: 40.0]                                                        │
│ --learning-rate          FLOAT    [default: 0.001]                                                       │
│ --epochs-per-task        INTEGER  [default: 5]                                                           │
│ --seed                   INTEGER  [default: 42]                                                          │
│ --fisher-samples         INTEGER  [default: 1000]                                                        │
│ --help                            Show this message and exit.                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.