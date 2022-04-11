import os
import torch

from torchvision import datasets, transforms

def load_training(root_path, directory, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         # transforms.RandomGrayscale(p=1),
         transforms.Normalize(mean=[0.47410455, 0.5279678, 0.77150476],
                              std=[0.39994287, 0.36881498, 0.1976388])
         ]
    )
    #data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    data = datasets.ImageFolder(root=os.path.join(root_path, directory), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def load_testing(root_path, directory, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         # transforms.RandomGrayscale(p=1),
         # transforms.Normalize(mean=[0.47410455, 0.5279678, 0.77150476],                   #cast
         #                      std=[0.39994287, 0.36881498, 0.1976388])
         transforms.Normalize(mean=[0.4625193, 0.5534448, 0.8095322],              #hit
                              std=[0.40904233, 0.35008422, 0.16984317])


         ]
    )
    #data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    data = datasets.ImageFolder(root=os.path.join(root_path, directory), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader