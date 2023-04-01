# Copyright (c) 2023 Bronte Sihan Li

# License: MIT License

import os
import torch
import pandas as pd
from sklearn.decomposition import PCA
from utils.train_test import criterion
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import numpy as np
from utils.net import Stock2DDataset, StockCNN

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


def load_data(test=False):
    # First, load the data.
    X_train = pd.read_csv(f'{PROJECT_PATH}/data/day10/train.csv', index_col=0)
    y_train = pd.read_csv(f'{PROJECT_PATH}/data/day10/train_labels.csv', index_col=0)
    X_val = pd.read_csv(f'{PROJECT_PATH}/data/day10/val.csv', index_col=0)
    y_val = pd.read_csv(f'{PROJECT_PATH}/data/day10/val_labels.csv', index_col=0)

    # Perform PCA on the data
    pca = PCA(n_components='mle', whiten=True, random_state=42)
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_val = pd.DataFrame(pca.transform(X_val))

    if test:
        X_test = pd.read_csv(f'{PROJECT_PATH}/data/day10/test.csv', index_col=0)
        y_test = pd.read_csv(f'{PROJECT_PATH}/data/day10/test_labels.csv', index_col=0)
        X_test = pd.DataFrame(pca.transform(X_test))
        test_dataset = Stock2DDataset(X_test, y_test)
        return test_dataset

    # Create the datasets
    train_dataset = Stock2DDataset(X_train, y_train)
    val_dataset = Stock2DDataset(X_val, y_val)

    return train_dataset, val_dataset


def train_cnn(config, checkpoint_dir='results/stock_cnn'):

    net = StockCNN(l1=config['l1'], l2=config['l2'], kernel_size=config['kernel_size'])
    device = 'cpu'
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_subset, val_subset = load_data()

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) / len(inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                for j, out in enumerate(outputs):
                    predicted = torch.gt(out, 0.5).float()
                    correct += (predicted == labels[j]).sum().item()
                total += labels.size(0) * len(out)
                loss = criterion(outputs, labels) / len(inputs)
                val_loss += loss.numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def test_accuracy(net, device="cpu"):
    testset = load_data(test=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False, num_workers=2
    )

    correct = 0
    with torch.no_grad():
        for data in testloader:
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            outputs = net(samples)
            for i, out in enumerate(outputs):
                predicted = torch.gt(out, 0.5).float()
                correct += (predicted == labels[i]).sum().item()

    n_labels = len(out)
    accuracy = correct / (len(testloader.dataset) * n_labels)
    return accuracy


if __name__ == '__main__':

    num_samples = 10
    max_num_epochs = 50
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 5)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 5)),
        "kernel_size": tune.choice([2, 3, 5]),
        "lr": tune.loguniform(1e-2, 1e-1),
        "momentum": tune.uniform(0.8, 0.9),
        "batch_size": tune.choice([16, 32, 64, 128]),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"]
    )
    result = tune.run(
        partial(train_cnn, checkpoint_dir='results/stock_cnn'),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 8},
        local_dir=f'{os.getcwd()}/results',
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )

    best_trained_model = StockCNN(
        best_trial.config["l1"],
        best_trial.config["l2"],
        best_trial.config["kernel_size"],
    )
    device = 'cpu'
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    print(best_checkpoint_dir)
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, "cpu")
    print("Best trial test set accuracy: {}".format(test_acc))
