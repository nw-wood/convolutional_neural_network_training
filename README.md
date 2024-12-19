# Fashion MNIST CNN Project

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify the Fashion MNIST dataset. It provides flexibility through command-line arguments for various hyperparameters such as learning rate, stride, padding and more.
A number of models will be produced upon running the train_evals.py file. They will be stored in the evaluations directory for inspection.

## Usage

Run the program from the terminal:

```bash
python train_eval.py
```

## Features

- **Dataset Loading**: Utilizes `torchvision.datasets` to fetch and preprocess the Fashion MNIST dataset.
- **Model Architecture**: Defines a simple CNN with configurable hyperparameters (kernel size, stride, padding, etc.).
- **Training & Testing**: Implements functions to train and evaluate the model, including accuracy reporting.
- **Command-Line Interface**: Allows users to adjust key hyperparameters without modifying the code.

## Requirements

- Python 3.7 or later
- Libraries:
  - `torch`
  - `torchvision`
  - `argparse`

Install dependencies using:
```bash
pip install torch torchvision
```

## Available Arguments - follows project guidelines by default

--num_workers	Number of subprocesses for data loading
--in_channels	Input channels for the first conv layer
--out_channels	Filters for the first conv layer
--kernel_size	Size of the convolution kernel
--stride	    Stride for convolution
--padding	    Padding for convolution
--lr	        Learning rate for the Adam optimizer

## File Structure

train_evals.py: Handles model training and evaluation.
dataset.py: Loads the Fashion MNIST dataset and prepares data loaders.
model.py: Defines the CNN architecture.
plot.py: Produces some graphs based on the trained models once completed.