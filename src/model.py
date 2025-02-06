import torch.nn as nn

class LecunModel(nn.Module):
    def __init__(self):
        super(LecunModel, self).__init__()

        # Feature extraction: convolution + pooling layers + fully connected
        self.features = nn.Sequential(
            # Conv layer 1: (1, 28, 28) -> (6, 26, 26)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),  # ReLU activation

            # MaxPool 1: (6, 26, 26) -> (6, 13, 13)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv layer 2: (6, 13, 13) -> (16, 11, 11)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),  # ReLU activation

            # MaxPool 2: (16, 11, 11) -> (16, 5, 5)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten 3D to 1D for fully connected layers
            nn.Flatten(),

            # Fully connected layers
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),  # Output layer (10 classes)
        )

    def forward(self, x):
        return self.features(x)  # Forward pass through the model
