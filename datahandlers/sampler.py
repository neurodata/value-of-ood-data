import torch
import numpy as np

from torch.utils.data import Sampler


class CustomBatchSampler(Sampler):
    """Samples 
    Provides equal representation of target classes in each batch
    """
    def __init__(self, cfg, tasks):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """

        self.cfg = cfg
        self.tasks = tasks
        self.npts = len(tasks)
        self.numtasks = len(cfg.task.ood) + 1
        self.num_iters = max(1, self.npts // cfg.hp.bs)

        # Indices for each task
        self.task_indices = []
        for i in range(self.numtasks):
            tinds = np.where(tasks==i)[0]
            np.random.shuffle(tinds)
            self.task_indices.append(tinds)

        self.β = cfg.task.beta
        if self.β == "unbiased":
            self.β = len(self.task_indices[0]) / len(tasks)

        # Num samples per task in each batch
        self.batch_num = []
        bs = cfg.hp.bs
        #   Target task samples per batch
        self.batch_num.append(max(1, int(self.β * bs)))

        #   OOD task samples per batch
        ood_frac = (1 - self.β) / (self.numtasks - 1)
        for i in range(1, self.numtasks):
            self.batch_num.append(int(ood_frac * bs))

    def __iter__(self):
        cfg = self.cfg

        # all_batches = []
        for it in range(self.num_iters):
            batch = []
            # sample 

            for i in range(self.numtasks):
                nind = len(self.task_indices[i])
                if nind == 0:
                    self.task_indices[i] = np.where(self.tasks==i)[0]
                    np.random.shuffle(self.task_indices[i])

                batch.extend(self.task_indices[i][:self.batch_num[i]])

                self.task_indices[i] = self.task_indices[i][self.batch_num[i]:]
            yield batch

    def __len__(self):
        return self.num_iters