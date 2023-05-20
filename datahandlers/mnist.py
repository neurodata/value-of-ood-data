import torchvision
import numpy as np
from datahandlers.basehandler import DatasetHandler
from datahandlers.domainbed_datasets import RotatedMNIST
import torchvision.transforms as transforms
import sys
import os
sys.path.insert(1, os.path.dirname(os.getcwd()))

class RotatedMNISTDataset(torchvision.datasets.VisionDataset):
    def __init__(self, angles, train=True, transform=None, target_transform=None):
        datasetholder = RotatedMNIST('../data', angles, train)
    
        data = []
        labels = []
        for i, angle in enumerate(angles):
            task_data, task_labels = datasetholder.datasets[i].tensors
            data.append(task_data.numpy().transpose(0, 2, 3, 1))
            labels.append([[i, lab] for lab in task_labels.numpy()])
        
        self.data = np.vstack(data)
        self.targets = np.concatenate(labels).astype('int').tolist()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

class RotatedMNISTHandler(DatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)

        mean_norm = [0.50]
        std_norm = [0.25]
        vanilla_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_norm, std=std_norm)])
        augment_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean_norm, std_norm)])

        if cfg.task.augment:
            train_transform = augment_transform
        else:
            train_transform = vanilla_transform
        test_transform = vanilla_transform

        angles = [0, cfg.task.angle]
        self.trainset = RotatedMNISTDataset(angles, train=True, transform=train_transform)
        self.testset = RotatedMNISTDataset([0], train=False, transform=test_transform)