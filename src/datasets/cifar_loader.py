from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets

def cifar_loader(batch_size=64, val_ratio=0.1):

    transform = transforms.ToTensor()

    full_train = datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=transform)
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    test_dataset = datasets.CIFAR10(root='../data/cifar', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader