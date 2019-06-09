import torch
from torchvision import datasets, transforms

def create_datasets(dataset, batch_size):
    if dataset == 'mnist':
        train_dset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        val_dset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        val_dset = datasets.CIFAR10('../data', train=False, transform=transform)
                    
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, val_loader