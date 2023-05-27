import torch.nn as nn
import wandb
import hydra
import torch
import numpy as np
import torch.cuda.amp as amp
import torchvision.transforms as transforms

from datahandlers.cinic import CINIC10
from torch.utils.data import DataLoader
from net.wideresnet import WideResNetSingleHeadNet
from net.smallconv import SmallConvSingleHeadNet
from utils.init import set_seed, open_log, init_wandb, cleanup

def get_data_loaders(cfg):
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

    mean_norm = [0.50, 0.50, 0.50]
    std_norm = [0.2, 0.25, 0.25]
    vanilla_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean_norm, std=std_norm)])
    augment_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean_norm, std_norm)])

    if cfg.augment:
        train_transform = augment_transform
    else:
        train_transform = vanilla_transform
    test_transform = vanilla_transform

    task = cfg.task_map[cfg.task]
    trainset = CINIC10('imagenet', task=task, flag='train', transform=train_transform)
    testset = CINIC10('imagenet', task=task, flag='test', transform=test_transform)

    kwargs = {
        'worker_init_fn': wif,
        'pin_memory': True,
        'num_workers': 4,
        'multiprocessing_context':'fork'
    }
    
    trainloader = DataLoader(trainset, batch_size=cfg.hp.bs, shuffle=True, **kwargs)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, **kwargs)

    return trainloader, testloader

def get_net(cfg):
    if cfg.net == 'wrn10_2':
        net = WideResNetSingleHeadNet(
            depth=10,
            num_cls=len(cfg.task_map[0]),
            base_chans=4,
            widen_factor=2,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'wrn16_4':
        net = WideResNetSingleHeadNet(
            depth=16,
            num_cls=len(cfg.task_map[0]),
            base_chans=16,
            widen_factor=4,
            drop_rate=0,
            inp_channels=3
        )
    elif cfg.net == 'conv':
        net = SmallConvSingleHeadNet(
            num_cls=len(cfg.task_map[0]),
            channels=1, # for cifar:3, mnist:80
            avg_pool=2,
            lin_size=80 # for cifar:320, mnist:80
        )
    else: 
        raise NotImplementedError

    return net

def train(cfg, net, trainloader, wandb_log=True):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    fp16 = device != 'cpu'
    net.to(device)

    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=cfg.hp.lr,
        momentum=0.9, 
        nesterov=True,
        weight_decay=1e-5
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        cfg.hp.epochs * len(trainloader)
    )
    scaler = amp.GradScaler()

    for epoch in range(cfg.hp.epochs):
        t_train_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        batches = 0.0

        criterion = nn.CrossEntropyLoss(reduction='none')
        net.train()
        for dat, labels in trainloader:
            labels = labels.long().to(device)
            dat = dat.to(device)
            batch_size = int(labels.size()[0])
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=fp16):
                out = net(dat)
                loss = criterion(out, labels).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # Compute Train metrics
            batches += batch_size
            train_loss += loss.item() * batch_size

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            train_acc += np.sum(labels == (np.argmax(out, axis=1)))

        info = {
            "epoch": epoch + 1,
            "train_loss": np.round(train_loss/batches, 4),
            "train_acc": np.round(train_acc/batches, 4)
        }
        print(info)

        if cfg.deploy and wandb_log:
            wandb.log(info)

    return net, loss, epoch, optimizer

def evaluate(cfg, net, testloader):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    net.to(device)

    net.eval()
    acc = 0
    count = 0

    with torch.no_grad():
        for dat, labels in testloader:
            dat = dat.to(device)
            labels = labels.long().to(device)
            batch_size = int(labels.size()[0])

            out = net(dat)

            out = out.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc += np.sum(labels == (np.argmax(out, axis=1)))
            count += batch_size

    error = 1 - (acc/count)
    info = {"final_test_err": error}

    print(info)

    if cfg.deploy:
        wandb.log(info)

    return error

def save_weights(cfg, net, loss, epoch, optimizer):
    torch.save(
    {
        'epoch':epoch,
        'model_state_dict':net.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss
    },
    "weights/pretrained_{}_{}_{}.pt".format(cfg.dataset, cfg.task, cfg.net)
)

@hydra.main(config_path="./config", config_name="pretrain_conf.yaml")
def main(cfg):
    init_wandb(cfg, project_name="ood_tl")
    fp = open_log(cfg)

    seed =  cfg.seed + 1 * 10
    set_seed(seed)
    net = get_net(cfg)
    trainloader, testloader = get_data_loaders(cfg)
    net, loss, epoch, optimizer = train(cfg, net, trainloader)
    err = evaluate(cfg, net, testloader)
    print("test error = ", err)
    save_weights(cfg, net, loss, epoch, optimizer)

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()
