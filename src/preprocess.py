from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

def load_data(batch_size=64, validation_split=0.2):
    """
    Load dataloaders for train, test and validation base
    :param batch_size: training batch size
    :param validation_split: validation data split
    :return: train, validation and test DataLoaders
    """
    train_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_data, valid_data = train_test_split(train_data, test_size=validation_split)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader