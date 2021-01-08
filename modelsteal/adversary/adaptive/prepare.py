#!/usr/bin/python
import modelsteal.utils.customized_model as model_utils
from torch import optim
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from modelsteal import datasets


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_optimizer(parameters, optimizer_type, lr=0.00001, momentum=0.5):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    sample_x = samples[0][0]
    if isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def train(model, trainset, testset, device=None):
    modelfamily = datasets.dataset_to_modelfamily['MNIST']
    transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    transferset = samples_to_transferset(trainset, transform=transform)
    optimizer = get_optimizer(model.parameters(), 'sgdm')
    criterion_train = model_utils.soft_cross_entropy
    model = model_utils.train_model(model, transferset, criterion_train=criterion_train, testset=testset, device=device, optimizer=optimizer)
    return model
