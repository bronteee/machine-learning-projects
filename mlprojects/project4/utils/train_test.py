# Copyright © 2023 "Bronte" Sihan Li

from typing import Literal
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import torch.optim as optim
from utils.net import Net


def load_data(
    batch_size_train: int,
    batch_size_test: int,
    dataset: str = Literal['mnist', 'fashion_mnist'],
    validation_split: float = None,
):
    """
    Loads the training and test data.
    """

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")

    tc_datasets = {
        'mnist': torchvision.datasets.MNIST,
        'fashion_mnist': torchvision.datasets.FashionMNIST,
    }

    trainset = tc_datasets[dataset](
        'files/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    if validation_split is not None:
        test_abs = int(len(trainset) * (1 - validation_split))
        train_subset, val_subset = random_split(
            trainset, [test_abs, len(trainset) - test_abs]
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size_train,
            shuffle=True,
            collate_fn=lambda x: [y.to(mps_device) for y in default_collate(x)],
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size_train,
            shuffle=True,
            collate_fn=lambda x: [y.to(mps_device) for y in default_collate(x)],
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size_train,
            shuffle=True,
            collate_fn=lambda x: [y.to(mps_device) for y in default_collate(x)],
        )
        val_loader = None

    test_loader = torch.utils.data.DataLoader(
        tc_datasets[dataset](
            'files/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
        collate_fn=lambda x: [y.to(mps_device) for y in default_collate(x)],
    )
    return train_loader, test_loader, val_loader


def train(
    epoch: int,
    train_loader: DataLoader,
    batch_size_train: int,
    network: Net,
    optimizer: optim.SGD,
    log_interval=10,
    save_dir: str = 'results',
):
    train_losses = []
    train_counter = []

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        network.to(mps_device)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        output = network(data.to(mps_device))
        target.to(mps_device)
        print(data.shape)
        print(output.shape, target.shape)
        # Compute negative log likelihood loss
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train)
                + ((epoch - 1) * len(train_loader.dataset))
            )
            # save the internal states
            torch.save(network.state_dict(), f'{save_dir}/model.pth')
            torch.save(optimizer.state_dict(), f'{save_dir}/optimizer.pth')

    return train_losses, train_counter


def test(network: Net, test_loader: DataLoader):

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        network.to(mps_device)

    test_losses = []
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # Sum up batch loss
            test_loss += F.nll_loss(
                F.log_softmax(output, dim=1), target, size_average=False
            ).item()
            # Get prediction
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
    print(
        '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_losses
