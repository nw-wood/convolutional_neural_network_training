import torch                            # Core PyTorch library for tensor computations and deep learning frameworks.
from torch import optim                 # Module providing various optimization algorithms (e.g., SGD, Adam).
from torch.autograd import Variable     # Facilitates computation of gradients for tensors.
import torch.nn as nn                   # Contains neural network layers, activations, and loss functions.
from dataset import FashionMNISTDataset # Custom dataset class for handling FashionMNIST data.
from model import CNN                   # Convolutional Neural Network architecture defined in the 'model' module.
import argparse                         # Module for parsing command-line arguments.
import os                               # Provides functions to interact with the operating system.

# Create the "evaluations" directory to store model outputs and metrics
os.makedirs('./evaluations', exist_ok=True)

# Determine the next model version number by checking existing saved models in the given path
def get_next_model_version(path):
    existing_models = [f for f in os.listdir(path) if f.startswith('model_') and f.endswith('.pt')]
    if not existing_models:
        return 1
    latest_version = max(int(f.split('_')[1].split('.')[0]) for f in existing_models)
    return latest_version + 1

# Train the model over a specified number of epochs, logging progress periodically
def train(num_epochs, cnn, loaders, loss_func, optimizer):
    cnn.train()
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)[0]  # Forward pass
            loss = loss_func(output, b_y)  # Compute loss

            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters

            if (i + 1) % 128 == 0:  # Log progress every 128 steps
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Evaluate the model on the test dataset and calculate accuracy
def test(cnn, loaders):
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, _ = cnn(images)  # Forward pass
            pred_y = torch.max(test_output, 1)[1].data.squeeze()  # Predicted labels
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total  # Compute accuracy
        print(f'Test Accuracy of the model: {accuracy * 100:.2f}%')
        return accuracy

# Save the trained model and its accuracy to a designated folder
def save_model(cnn, model_name, folder_path, accuracy):
    version = get_next_model_version(folder_path)
    model_name = f'model_{version}.pt' if model_name is None else model_name
    model_path = f'{folder_path}/{model_name}'
    torch.save(cnn.state_dict(), model_path)  # Save the model's state
    print(f'Model saved to {model_path}')

    # Save accuracy alongside the model
    accuracy_file = f'{folder_path}/{model_name.replace(".pt", "")}_accuracy.txt'
    with open(accuracy_file, 'w') as file:
        file.write(f'{model_name}: {accuracy * 100:.2f}%\n')

# Main function to configure arguments and manage the training workflow
def main():

    # Define configurable parameters for the training script
    parser = argparse.ArgumentParser(description="Train a CNN on FashionMNIST")
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels for the model')
    parser.add_argument('--out_channels', type=int, default=8, help='Number of output channels for the first conv layer')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for conv layers')
    parser.add_argument('--stride', type=int, default=1, help='Stride for conv layers')
    parser.add_argument('--padding', type=int, default=0, help='Padding for conv layers')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for Adam optimizer')

    # Parse the arguments provided via the command line
    args = parser.parse_args()

    # Initialize the CNN model with the specified hyperparameters
    cnn = CNN(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding
    )
    print(cnn)

    # Define the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=args.lr)

    # Define hyperparameter ranges for training multiple models
    epochs_set = [5, 10, 15, 20, 25, 30]  # Varying number of training epochs
    batch_size_set = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Varying batch sizes

    # Iterate over all combinations of epochs and batch sizes
    for epochs in epochs_set:
        for batch_size in batch_size_set:

            # Create a subdirectory to store the results of each combination
            folder_path = f'./evaluations/epochs_{epochs}_batch_size_{batch_size}'
            os.makedirs(folder_path, exist_ok=True)

            print(f"Training model with {epochs} epochs and batch size {batch_size}...")

            # Initialize the dataset and data loaders with the current batch size
            dataset = FashionMNISTDataset(batch_size=batch_size, num_workers=args.num_workers)
            loaders = dataset.load_data()

            # Reinitialize the CNN model for each training iteration
            cnn = CNN(in_channels=args.in_channels, out_channels=args.out_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.padding)
            optimizer = optim.Adam(cnn.parameters(), lr=args.lr)

            # Train the model and evaluate its accuracy
            train(epochs, cnn, loaders, loss_func, optimizer)
            accuracy = test(cnn, loaders)

            # Save the trained model and its accuracy
            save_model(cnn, model_name=None, folder_path=folder_path, accuracy=accuracy)

# Entry point for the script execution
if __name__ == "__main__":
    main()
