from copy import deepcopy
import torchvision
import numpy as np
from datahandlers.basehandler import DatasetHandler
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import ShuffleSplit
from datahandlers.domainbed_datasets import DomainNet
import deeplake

class DomainNetDataset(torchvision.datasets.VisionDataset):
    def __init__(self, envs, classes, train=True, transform=None, target_transform=None):
        ENVIRONMENTS = ["paint", "quick", "real", "sketch"]
        datasetholder = DomainNet('/cis/home/adesilva/ashwin/research/ood-tl/data', envs, train)
        
        imgs = []
        labels = []
        if train:
            envset = [ENVIRONMENTS.index(envs[0]), ENVIRONMENTS.index(envs[1])]
        else:
            envset = [0] # test is always real
        for i, env in enumerate(envset):
            task_imgs, task_labels = datasetholder.datasets[env].imgs, datasetholder.datasets[env].targets
            imgs.append(task_imgs)
            labels.append(np.array([[i, lab] for lab in task_labels]))
        
        labels = np.concatenate(labels)
        imgs = np.concatenate(imgs)

        indices = []
        for idx, label in enumerate(labels):
            if label[1] in classes:
                indices.append(idx)
        
        imgs = imgs[indices]
        labels = labels[indices]
        cls_dict = dict(zip(classes, np.arange(0, len(classes)).tolist()))
        labels = [[lab[0], cls_dict[lab[1]]] for lab in labels]

        self.data = imgs
        self.targets = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_path, target = self.data[idx][0], self.targets[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

class DomainNetHandler(DatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)

        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.20, 0.25, 0.25]
        vanilla_transform = transforms.Compose([
                            transforms.Resize((64,64)),
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

        envs = [str(cfg.task.target_env), str(cfg.task.ood_env)]
        classes = cfg.task.task_map[0]
        self.trainset = DomainNetDataset(envs, classes, train=True, transform=train_transform)
        self.testset = DomainNetDataset([envs[0]], classes, train=False, transform=test_transform)