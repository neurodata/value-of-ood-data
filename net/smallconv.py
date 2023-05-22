import torch
import torch.nn as nn

# define the base CNN
class SmallConvSingleHeadNet(nn.Module):
    """
    Small convolution network with no residual connections (single-head)
    """
    def __init__(self, num_cls=10, channels=3, avg_pool=2, lin_size=320):
        super(SmallConvSingleHeadNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size
        self.fc = nn.Linear(self.linsize, num_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.flatten(1, -1)

        x = self.fc(x)
        return x

class SmallConvMultiHeadNet(nn.Module):
    """
    Small convolution network with no residual connections (multi-head)
    """
    def __init__(self, num_task=1, num_cls=10, channels=3, avg_pool=2, lin_size=320):
        super(SmallConvMultiHeadNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size

        lin_layers = []
        for task in range(num_task):
            lin_layers.append(nn.Linear(self.linsize, num_cls)) # add fully connected layers for each task

        self.fc = nn.ModuleList(lin_layers) # holds the task specific FC 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, tasks):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.view(-1, self.linsize)

        logits = self.fc[0](x) * 0 # get a zero-vector

        for idx, lin in enumerate(self.fc):
            task_idx = torch.nonzero((idx == tasks), as_tuple=False).view(-1) # select the training examples in the batch that belongs to the current task
            if len(task_idx) == 0: # if there are no training examples for the current task, continue
                continue

            task_out = torch.index_select(x, dim=0, index=task_idx) # obtain the training examples of the current task
            task_logit = lin(task_out) # task-specific FC layer
            logits.index_add_(0, task_idx, task_logit) # add the task-specific logits to the full-logits vector

        return logits
