import torch.nn as nn  # Defines neural network modules and layers.

class CNN(nn.Module):
    """
    Defines a Convolutional Neural Network (CNN) for FashionMNIST classification.

    This CNN consists of two convolutional layers, each followed by ReLU activation
    and max-pooling layers, and a final fully connected layer to classify the input
    into one of 10 categories (digits 0-9).

    Args:
        in_channels (int): Number of input channels for the first convolutional layer (e.g., 1 for grayscale images).
        out_channels (int): Number of output channels for the first convolutional layer (controls feature maps).
        kernel_size (int): Size of the kernel used in convolutional layers (e.g., 3x3 or 5x5).
        stride (int): Step size for the convolution operation (controls downsampling).
        padding (int): Number of zero-padding pixels added to the input edges (prevents size reduction during convolution).
    """
    def __init__(self, in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0):
        super(CNN, self).__init__()
        # First convolutional layer
        # Input -> [Batch, in_channels, Height, Width]
        # Output -> [Batch, out_channels, H_out, W_out]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,     # Number of input channels (grayscale images = 1 channel)
                out_channels=out_channels,   # Number of feature maps produced by this layer
                kernel_size=kernel_size,     # Size of the convolutional filter (e.g., 5x5)
                stride=stride,               # Step size for the convolution
                padding=padding,             # Padding to preserve spatial dimensions
            ),
            nn.ReLU(),                       # Non-linear activation function to introduce learning capacity
            nn.MaxPool2d(kernel_size=2),     # Downsample by taking the max value in 2x2 regions
        )

        # Second convolutional layer
        # Doubles the number of feature maps produced by conv1
        # Input -> [Batch, out_channels, H_out, W_out]
        # Output -> [Batch, out_channels * 2, H_out/2, W_out/2]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,    # Takes feature maps from conv1
                out_channels=out_channels * 2,  # Doubles the number of feature maps
                kernel_size=kernel_size,     # Same kernel size as the first layer
                stride=stride,               # Same stride as the first layer
                padding=padding,             # Same padding as the first layer
            ),
            nn.ReLU(),                       # Non-linear activation function
            nn.MaxPool2d(kernel_size=2),     # Downsample further with a 2x2 pooling operation
        )

        # Fully connected output layer
        # Input size depends on the output size of the second convolutional layer
        # Here, it's calculated as (out_channels * 2 * 4 * 4), assuming input image size is 28x28
        # Output -> [Batch, 10] (10 classes for classification)
        self.out = nn.Linear(out_channels * 2 * 4 * 4, 10)

    def forward(self, x):
        """
        Defines the forward pass of the CNN. This method takes the input tensor `x`
        and processes it through the layers defined in the constructor.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, in_channels, Height, Width].

        Returns:
            tuple: 
                - Output predictions of shape [Batch, 10] (logits for classification).
                - Flattened feature representation of shape [Batch, Features].
        """
        x = self.conv1(x)  # Pass through the first convolutional layer
        x = self.conv2(x)  # Pass through the second convolutional layer
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        output = self.out(x)  # Pass through the fully connected layer
        return output, x  # Return both predictions and flattened features
