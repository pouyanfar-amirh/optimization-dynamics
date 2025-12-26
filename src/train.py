import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from src.models.MNISTSimpleCNN import MNISTSimpleCNN
from src.models.CIFARSimpleCNN import CIFARSimpleCNN
from src.datasets.mnist_loader import mnist_loader
from src.datasets.cifar_loader import cifar_loader
from src.utils import plot_loss

def train_SimpleCNN(num_epochs=10, batch_size=32, optimization='SGD', learning_rate=0.001, dataset='mnist'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset == 'mnist':
        train_loader, val_loader, test_loader = mnist_loader(batch_size=batch_size)
        model = MNISTSimpleCNN(input_channels=1, num_classes=10).to(device)
    elif dataset == 'cifar':
        train_loader, val_loader, test_loader = cifar_loader(batch_size=batch_size)
        model = CIFARSimpleCNN(input_channels=3, num_classes=10).to(device)
    else:
        raise ValueError('Invalid dataset')

    
    criterion = nn.CrossEntropyLoss()

    optimization = optimization.lower()
    if optimization == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimization == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), learning_rate)
    elif optimization == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimization == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimization type')


    loss_values = []

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        total_examples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #print(f'Output shape : {outputs.shape}')

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
        

        epoch_loss = total_loss / total_examples
        loss_values.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/total_examples:.4f}')

    plot_loss(loss_values)

    os.makedirs('../results', exist_ok=True)
    torch.save(model.state_dict(), f'../results/SimpleCNN_{dataset.lower}.pth')