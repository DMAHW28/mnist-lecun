import torch.nn as nn

class LecunModel(nn.Module):
    def __init__(self):
        super(LecunModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1),  # (1, 32, 32) to (6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (6, 28, 28) to (6, 14, 14)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1),  # (6, 14, 14) to (16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 10, 10) to (16, 5, 5)
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        y = self.features(x)
        return y
