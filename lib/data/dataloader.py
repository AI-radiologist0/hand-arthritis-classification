# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(data_path, batch_size, split_ratio):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)

    train_size = int(split_ratio['train'] * len(dataset))
    val_size = int(split_ratio['validation'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
