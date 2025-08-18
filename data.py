import torchvision as tv
from torch.utils.data import DataLoader

def get_MNIST(batch_size: int) -> DataLoader:
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5),  # [-1,1] for all channels if grayscale; for RGB use per-channel tuples
    ])

    dataset = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return dataloader

def get_FashionMNIST(batch_size: int) -> DataLoader:
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5),  # [-1,1] for all channels if grayscale; for RGB use per-channel tuples
    ])

    dataset = tv.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return dataloader
