import sys
import os
import torch
import argparse
import numpy as np
from torch import nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import LecunModel # Import the model class
from src.preprocess import load_data # Import the data loading function

class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0

    def init_params(self):
        # Reset training and validation statistics
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0

    def train_step(self, data, criterion, optimizer):
        # Perform a single training step (forward + backward pass)
        self.model.train()
        X, y = data
        optimizer.zero_grad()
        y_pred = self.model(X)
        # Compute loss
        loss = criterion(y_pred, y)
        self.train_loss += loss.item()
        # Get predicted class
        pred = y_pred.argmax(dim=1, keepdim=True)
        self.train_acc += pred.eq(y.view_as(pred)).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()

    def val_step(self, data, criterion):
        # Perform a validation step
        self.model.eval()
        X, y = data
        y_pred = self.model(X)
        # Compute loss
        loss = criterion(y_pred, y)
        self.val_loss += loss.item()
        # Get predicted class
        pred = y_pred.argmax(dim=1, keepdim=True)
        self.val_acc += pred.eq(y.view_as(pred)).sum().item()


def train_model(epochs = 10, batch_size = 64, learning_rate = 1e-2, validation_split=0.2):
    # Load the MNIST dataset
    train_loader, valid_loader, _ = load_data(batch_size=batch_size, validation_split=validation_split)

    # Initialize the model and training components
    mnist_model = LecunModel()
    criterion = nn.CrossEntropyLoss()
    sgd = optim.SGD(mnist_model.parameters(), lr=learning_rate)
    epochs = epochs
    trainer = Trainer(mnist_model)
    # Lists to store the training and validation statistics
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []

    # Training loop
    for epoch in range(epochs):
        # Set model to training mode and reset statistics at the start of each epoch
        mnist_model.train()
        trainer.init_params()
        for data_train in train_loader:
            # Perform training steps
            trainer.train_step(data_train, criterion, sgd)
        # Store training stats for this epoch
        train_losses.append(trainer.train_loss / len(train_loader))
        train_acc.append(trainer.train_acc / len(train_loader.dataset))
        with torch.no_grad():
            for data_val in valid_loader:
                # Perform validation steps
                trainer.val_step(data_val, criterion)
            # Store validation stats for this epoch
            valid_losses.append(trainer.val_loss / len(valid_loader))
            valid_acc.append(trainer.val_acc / len(valid_loader.dataset))

    print("Training completed.")
    # Save training and validation statistics to a file
    np.savez("./stats/lenet_mnist_stats.npz", train_losses=np.array(train_losses), train_acc=np.array(train_acc), valid_losses=np.array(valid_losses), valid_acc=np.array(valid_acc), allow_pickle=True)
    # Save the trained model parameters
    torch.save(mnist_model.state_dict(), "./models/lenet_mnist.pth")
    print("Model saved to models/lenet_mnist.pth")

if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train the LeNet-5 model on MNIST dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--val_split", type=float, default=0.2, help="Split portion for validation base")
    args = parser.parse_args()

    # Train the model with the provided arguments
    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, validation_split=args.val_split)
