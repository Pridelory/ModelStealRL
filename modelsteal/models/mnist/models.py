import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = ['lenet', 'resnet']


class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def lenet(num_classes, **kwargs):
    return LeNet(num_classes, **kwargs)

class Reset(nn.Module):
    def __init__(self, pretrained, output_classes=10, **kwargs):
        super(Reset, self).__init__()
        # self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.reset = torchvision.models.resnet18(pretrained)
        self.fc = nn.Linear(1000, output_classes)

    def forward(self, x):
        # x = self.conv(x)
        x = self.reset(x)
        x = self.fc(x)
        x = F.softmax(x)
        return x

def resnet(num_classes, **kwargs):
    return Reset(num_classes, **kwargs)

