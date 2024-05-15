from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    return train_loader
