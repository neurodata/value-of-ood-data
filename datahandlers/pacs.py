from copy import deepcopy
import torchvision
import numpy as np
from datahandlers.basehandler import DatasetHandler
from datahandlers.domainbed_datasets import OfficeHome
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import ShuffleSplit
from datahandlers.domainbed_datasets import PACS

class PACSDataset(torchvision.datasets.VisionDataset):
    def __init__(self, envs, classes, train=True, transform=None, target_transform=None):
        ENVIRONMENTS = ["A", "C", "P", "S"]
        datasetholder = PACS('/cis/home/adesilva/ashwin/research/ood-tl/data', envs)
    
        imgs = []
        labels = []
        envset = [ENVIRONMENTS.index(envs[0]), ENVIRONMENTS.index(envs[1])]
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

        self.data = imgs.tolist()
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

class PACSHandler(DatasetHandler):
    def __init__(self, cfg):
        super().__init__(cfg)

        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.20, 0.25, 0.25]
        # mean_norm =[0.485, 0.456, 0.406]
        # std_norm =[0.229, 0.224, 0.225]
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
        dataset = PACSDataset(envs, classes, transform=train_transform)

        data = np.array(dataset.data)
        labels = np.array(dataset.targets)
        indist_indices = np.where(labels[:, 0]==0)[0]
        indist_labels = labels[indist_indices]
        tr_indices, te_indices = next(ShuffleSplit(n_splits=1, random_state=0, test_size=0.75).split(np.zeros(indist_labels.shape), indist_labels[:, 1]))

        indist_data = data[indist_indices]
        indist_traindata, indist_trainlabels = indist_data[tr_indices], indist_labels[tr_indices]
        indist_testdata, indist_testlabels = indist_data[te_indices], indist_labels[te_indices]

        outdist_indices = np.where(labels[:, 0]==1)[0]
        outdist_traindata, outdist_trainlabels = data[outdist_indices], labels[outdist_indices]

        trainset = deepcopy(dataset)
        testset = deepcopy(dataset)

        trainset.data = np.concatenate((indist_traindata, outdist_traindata))
        trainset.targets = np.concatenate((indist_trainlabels, outdist_trainlabels)).tolist()

        testset.data = indist_testdata
        testset.targets = indist_testlabels.tolist()
        testset.transform = test_transform

        self.trainset = trainset
        self.testset = testset

