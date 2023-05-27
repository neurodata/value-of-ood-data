import hydra
import torch
import torchvision
import numpy as np
import wandb
import pandas as pd

from utils.init import set_seed, open_log, init_wandb, cleanup

from datahandlers.cifar import SplitCIFARHandler, RotatedCIFAR10Handler, BlurredCIFAR10Handler
from datahandlers.cinic import SplitCINIC10Handler, SplitCIFAR10NegHandler
from datahandlers.mnist import RotatedMNISTHandler
from datahandlers.officehomes import OfficeHomeHandler
from datahandlers.pacs import PACSHandler
from net.smallconv import SmallConvSingleHeadNet, SmallConvMultiHeadNet
from net.wideresnet import WideResNetSingleHeadNet, WideResNetMultiHeadNet

from utils.run_net import train, evaluate
from utils.tune import search_alpha


def get_data(cfg, seed):
    if cfg.task.dataset == "split_cifar10":
        dataHandler = SplitCIFARHandler(cfg)
    elif cfg.task.dataset == "split_cinic10":
        dataHandler = SplitCINIC10Handler(cfg)
    elif cfg.task.dataset == "rotated_cifar10":
        dataHandler = RotatedCIFAR10Handler(cfg)
    elif cfg.task.dataset == "blurred_cifar10":
        dataHandler = BlurredCIFAR10Handler(cfg)
    elif cfg.task.dataset == "split_cifar10neg":
        dataHandler = SplitCIFAR10NegHandler(cfg)
    elif cfg.task.dataset == "rotated_mnist":
        dataHandler = RotatedMNISTHandler(cfg)
    elif cfg.task.dataset == "officehomes":
        dataHandler = OfficeHomeHandler(cfg)
    elif cfg.task.dataset == "pacs":
        dataHandler = PACSHandler(cfg)
    else:
        raise NotImplementedError

    # Use different seeds across different runs
    # But use the same seed
    dataHandler.sample_data(seed)
    task_labels = np.array(dataHandler.comb_trainset.targets)[:, 0]
    num_target_samples = len(task_labels[task_labels==0])
    num_ood_samples = len(task_labels[task_labels==1])
    info = {
        "n": num_target_samples,
        "m": num_ood_samples
    }
    if cfg.deploy:
        wandb.log(info)
    trainloader = dataHandler.get_data_loader(train=True)
    testloader = dataHandler.get_data_loader(train=False)
    unshuffled_trainloader = dataHandler.get_data_loader(train=True, shuffle=False)
    return trainloader, testloader, unshuffled_trainloader


def get_net(cfg):
    if cfg.net == 'wrn10_2':
        net = WideResNetSingleHeadNet(
            depth=10,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=4,
            widen_factor=2,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'wrn16_4':
        net = WideResNetSingleHeadNet(
            depth=16,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=16,
            widen_factor=4,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'conv':
        net = SmallConvSingleHeadNet(
            num_cls=len(cfg.task.task_map[0]),
            channels=1, # for cifar:3, mnist:1
            avg_pool=2,
            lin_size=80 # for cifar:320, mnist:80
        )
    elif cfg.net == 'multi_conv':
        net = SmallConvMultiHeadNet(
            num_task=2,
            num_cls=len(cfg.task.task_map[0]),
            channels=3, 
            avg_pool=2,
            lin_size=320
        )
    elif cfg.net == 'multi_wrn10_2':
        net = WideResNetMultiHeadNet(
            depth=10,
            num_task=2,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=4,
            widen_factor=2,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'multi_wrn16_4':
        net = WideResNetMultiHeadNet(
            depth=16,
            num_task=2,
            num_cls=len(cfg.task.task_map[0]),
            base_chans=16,
            widen_factor=4,
            drop_rate=0.2,
            inp_channels=3
        )
    else: 
        raise NotImplementedError

    return net

@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    init_wandb(cfg, project_name="ood_tl")
    fp = open_log(cfg)

    opt_alpha_list = [1]
    opt_err_list = [0.5]
    if cfg.loss.tune_alpha:
        for m_n in cfg.loss.m_n_list:
            if m_n == 0:
                continue
            seed =  cfg.seed
            set_seed(seed)
            net = get_net(cfg)
            cfg.task.m_n = m_n
            dataloaders = get_data(cfg, seed)
            opt_alpha, opt_err = search_alpha(cfg, opt_alpha_list[-1], net, dataloaders)
            opt_alpha_list.append(opt_alpha)
            opt_err_list.append(opt_err)
            info = {
                "opt_alpha_list": opt_alpha_list,
                "opt_err_list": opt_err_list
            }
            if cfg.deploy:
                wandb.log(info)

if __name__ == "__main__":
    main()
