import torch
from src.models.SimpleCNN import SimpleCNN
from src.datasets import mnist_loader
import torch.nn.functional as F

def evaluate_mnist():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = mnist_loader(batch_size=32)

    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load('results/SimpleCNN.pth'))
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
