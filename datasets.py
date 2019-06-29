from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataset(name, batch_size):
    if name == 'MNIST':
        return DataLoader(datasets.MNIST('data/mnist', download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])), batch_size=batch_size, shuffle=True)
