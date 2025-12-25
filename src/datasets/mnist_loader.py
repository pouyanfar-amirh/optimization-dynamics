from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def mnist_loader(batch_size=64, val_ratio=0.1):
    transform = transforms.ToTensor()
    
    full_train = datasets.MNIST(root='../data/mnist', download=True, train=True, transform=transform)
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    test_dataset = datasets.MNIST(root="../data/mnist", download=True, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader