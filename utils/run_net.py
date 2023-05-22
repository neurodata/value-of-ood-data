import torch.nn as nn
import wandb
import torch
import numpy as np
import torch.cuda.amp as amp


def train(cfg, net, trainloaders, wandb_log=True):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    fp16 = device != 'cpu'
    if cfg.ptw:
        checkpoint = torch.load('weights/' + cfg.ptw_path)
        net.load_state_dict(checkpoint['model_state_dict']) # load the imagenet pretrained weights
        net.fc = nn.Linear(net.fc.in_features, len(cfg.task.task_map[0])) # randomly initialize the linear layer
        
        # freeze the network except for the linear layer
        for param in net.parameters():
            param.requires_grad = False
        net.fc.bias.requires_grad = True
        net.fc.weight.requires_grad = True
    
    net.to(device)

    trainloader = trainloaders[0]
    unshuffled_trainloader = trainloaders[2]

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
    ntasks = len(cfg.task.ood) + 1

    num_trainsamples = len(unshuffled_trainloader.dataset)
    epoch_agreement_matrix = np.zeros((num_trainsamples, cfg.hp.epochs))
    epoch_proba_matrix = np.zeros((num_trainsamples, cfg.hp.epochs))

    for epoch in range(cfg.hp.epochs):
        t_train_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        batches = 0.0

        if cfg.ptw and (epoch > 50):
            # above a certain epoch, train with a fully unfreezed network
            for param in net.parameters():
                param.requires_grad = True
    
        criterion = nn.CrossEntropyLoss(reduction='none')
        net.train()

        for dat, target in trainloader:
            tasks, labels = target
            tasks = tasks.long().to(device)
            labels = labels.long().to(device)
            dat = dat.to(device)
            batch_size = int(labels.size()[0])
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=fp16):
                if cfg.is_multihead:
                    out = net(dat, tasks)
                else:
                    out = net(dat)

                if cfg.loss.group_task_loss:
                    
                    # loss = criterion(out, labels)
                    # wt = cfg.loss.alpha
                    # wo = (1-cfg.loss.alpha)
                    # loss_target = torch.nan_to_num(loss[tasks==0].mean())
                    # loss_ood = torch.nan_to_num(loss[tasks==1].mean())
                    # final_loss = wt*loss_target + wo*loss_ood
                    
                    task_oh = torch.nn.functional.one_hot(tasks, ntasks)
                    task_count = task_oh.sum(0)

                    loss = criterion(out, labels)
                    task_loss = (loss.view(-1, 1) * task_oh).sum(0)

                    mask = task_count != 0
                    task_loss[mask] /= task_count[mask]

                    task_loss[1:] *=  (1 - cfg.loss.alpha) / (ntasks - 1)
                    task_loss[0] *= (cfg.loss.alpha)

                    final_loss = task_loss.sum()

                else:
                    final_loss = criterion(out, labels).mean()

            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # Compute Train metrics
            batches += batch_size
            train_loss += final_loss.item() * batch_size

            if cfg.loss.group_task_loss:
                tl = task_loss.detach().to('cpu').numpy()
                t_train_loss += tl * batch_size  # This is not exact

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            train_acc += np.sum(labels == (np.argmax(out, axis=1)))

        # evaluate the network on all the training data at the end of each epoch
        if cfg.eval_at_epoch:
            net.eval()
            with torch.no_grad():
                pred_agreement = []
                pred_proba = []
                task_ids = []
                for dat, target in unshuffled_trainloader:
                    tasks, labels = target
                    dat = dat.to(device)
                    tasks = tasks.long().to(device)
                    labels = labels.long().to(device)

                    out = net(dat)
                    out = nn.functional.softmax(out)
                    out = out.cpu().detach().numpy()
                    labels = labels.cpu().numpy()
                    pred_agreement.extend(list(labels == (np.argmax(out, axis=1))))
                    pred_proba.extend(list(np.max(out, axis=1)))
                    task_ids.extend(list(tasks.cpu().numpy()))

            epoch_agreement_matrix[:, epoch] = np.array(pred_agreement).astype('int')
            epoch_proba_matrix[:, epoch] = np.around(np.array(pred_proba), decimals=5).astype(np.float16)

            info = {
                "epoch": epoch + 1,
                "train_loss": np.round(train_loss/batches, 4),
                "train_acc": np.round(train_acc/batches, 4),
                "task_ids": task_ids,
                "epoch_agreement_matrix": epoch_agreement_matrix.tolist(),
                "epoch_proba_matrix": epoch_proba_matrix.tolist()
            }
            
        else:
            info = {
                "epoch": epoch + 1,
                "train_loss": np.round(train_loss/batches, 4),
                "train_acc": np.round(train_acc/batches, 4)
            }

        if cfg.deploy and wandb_log:
            wandb.log(info)

        if cfg.loss.group_task_loss:
            info["task_loss"] = tuple(np.round(t_train_loss/batches, 4))

        if train_acc/batches >= 0.99:
            break
    
    return net


def evaluate(cfg, net, testloader, run_num):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    net.to(device)

    net.eval()
    acc = 0
    count = 0

    with torch.no_grad():
        for dat, target in testloader:

            tasks, labels = target
            dat = dat.to(device)
            tasks = tasks.long().to(device)
            labels = labels.long().to(device)
            batch_size = int(labels.size()[0])

            if cfg.is_multihead:
                out = net(dat, tasks)
            else:
                out = net(dat)

            out = out.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc += np.sum(labels == (np.argmax(out, axis=1)))
            count += batch_size

    error = 1 - (acc/count)
    info = {"run_num": run_num,
            "final_test_err": error}
    # print(info)
    if cfg.deploy:
        wandb.log(info)

    return error
