import torch
from src.models.MNISTSimpleCNN import MNISTSimpleCNN
from src.models.CIFARSimpleCNN import CIFARSimpleCNN
from src.datasets.mnist_loader import mnist_loader
from src.datasets.cifar_loader import cifar_loader
import torch.nn.functional as F

def evaluate_SimpleCNN_test(dataset='mnist', batch_size=32):

    if dataset != 'mnist' and dataset != 'cifar':
        raise ValueError('Invalid dataset for evaluating the model on test set')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset == 'mnist':
        _, _, test_loader = mnist_loader(batch_size=batch_size)
        model = MNISTSimpleCNN(input_channels=1, num_classes=10).to(device)
    else:
        _, _, test_loader = cifar_loader(batch_size=batch_size)
        model = CIFARSimpleCNN(input_channels=3, num_classes=10).to(device)

    
    model.load_state_dict(torch.load(f'../results/SimpleCNN_{dataset}.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Test accuracy = {accuracy * 100:.2f}%')

