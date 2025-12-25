import matplotlib.pyplot as plt
import os

def plot_loss(loss_values, title='Training Loss', xlabel='Epoch', ylabel='Loss', save_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(loss_values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    