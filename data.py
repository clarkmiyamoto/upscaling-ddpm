import torchvision as tv
from torch.utils.data import DataLoader

def get_data(dataset: str, batch_size: int, num_workers: int = 4) -> tuple[DataLoader, DataLoader, int]:
    '''
    Get the dataloader for the dataset.
    Args:
        dataset (str): the name of the dataset
        batch_size (int): the batch size
        num_workers (int): the number of workers
    Returns:
        dataloader_train (DataLoader): the dataloader for the training set
        dataloader_val (DataLoader): the dataloader for the validation set
        channels (int): the number of channels in the dataset
    '''
    if dataset == "MNIST":
        return get_MNIST(batch_size, num_workers)
    elif dataset == "FashionMNIST":
        return get_FashionMNIST(batch_size, num_workers)
    elif dataset == "CelebA":
        return get_CelebA(batch_size, num_workers)

def get_MNIST(batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5),  # [-1,1] for all channels if grayscale; for RGB use per-channel tuples
    ])

    dataset_train = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataset_val = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    channels = 1
    
    return dataloader_train, dataloader_val, channels

def get_FashionMNIST(batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5),  # [-1,1] for all channels if grayscale; for RGB use per-channel tuples
    ])

    dataset_train = tv.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    dataset_val = tv.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    channels = 1
    
    return dataloader_train, dataloader_val, channels

def get_CelebA(batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset_train = tv.datasets.CelebA(root="./data", split="train", download=True, transform=transform)
    dataset_val = tv.datasets.CelebA(root="./data", split="valid", download=True, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    channels = 3
    
    return dataloader_train, dataloader_val, channels
