import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.transform import rotate
from skimage.filters import gaussian

from typing import List
from copy import deepcopy

from datahandlers.sampler import CustomBatchSampler

class DatasetHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainset = None
        self.testset = None

    def sample_data(self, seed):
        ## Balanced sample for each data

        cfg = self.cfg
        comb_trainset = deepcopy(self.trainset)
        data = self.trainset.data
        targets = np.array(self.trainset.targets)
        tasks = targets[:, 0]

        num_tasks = len(cfg.task.ood) + 1
        indices = []

        for i in range(num_tasks):
            idx = (np.where(tasks == i))[0]
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

            nsamples = cfg.task.n
            if i > 0:
                # nsamples *= cfg.task.m_n 
                nsamples = cfg.task.m_n # passing m through m_n
            nsamples = int(nsamples)

            if nsamples > 0:
                for lb in range(len(cfg.task.task_map[0])):
                    lab_idx = np.where(targets[idx, 1] == lb)[0]
                    if i > 0: # for rotated and blurred CIFAR-10, we don't want to target indices appearing in the OOD set
                        indices.extend(list(idx[lab_idx][cfg.task.n:cfg.task.n + nsamples]))
                    else:
                        indices.extend(list(idx[lab_idx][:nsamples]))

        comb_trainset.data = data[indices]
        comb_trainset.targets = targets[indices].tolist()
        self.comb_trainset = comb_trainset

    def get_data_loader(self, train=True, shuffle=True):
        def wif(id):
            """
            Used to fix randomization bug for pytorch dataloader + numpy
            Code from https://github.com/pytorch/pytorch/issues/5059
            """
            process_seed = torch.initial_seed()
            # Back out the base_seed so we can use all the bits.
            base_seed = process_seed - id
            ss = np.random.SeedSequence([id, base_seed])
            # More than 128 bits (4 32-bit words) would be overkill.
            np.random.seed(ss.generate_state(4))

        cfg = self.cfg

        kwargs = {
            'worker_init_fn': wif,
            'pin_memory': True,
            'num_workers': 4,
            'multiprocessing_context':'fork'}

        if train:
            if cfg.task.custom_sampler and cfg.task.m_n > 0:
                tasks = np.array(self.comb_trainset.targets)[:, 0]
                batch_sampler = CustomBatchSampler(cfg, tasks)

                data_loader = DataLoader(
                    self.comb_trainset, batch_sampler=batch_sampler, **kwargs)
            else:
                # If no OOD samples use naive sampler
                data_loader = DataLoader(
                    self.comb_trainset, batch_size=cfg.hp.bs,
                    shuffle=shuffle, **kwargs)
        else:
            data_loader = DataLoader(
                self.testset, batch_size=cfg.hp.bs, shuffle=False, **kwargs)
                

        return data_loader
