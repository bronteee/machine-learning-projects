import torch
import torch.nn as nn
from utils.train_test import train, test
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X.iloc[idx].to_numpy()).float()
        y = torch.from_numpy(self.y.iloc[idx].to_numpy()).float()
        if self.transform:
            x = self.transform(x)
        return x, y


class StockLinearNN(nn.Module):
    """
    Defines the neural network architecture for the linear model
    for the stock dataset.
    """

    def __init__(
        self,
        l2: int = 3,  # number of nodes in the second fully connected layer
        l3: int = 3,  # number of nodes in the third fully connected layer
    ):
        super(StockLinearNN, self).__init__()
        self.l2 = l2
        self.l3 = l3
        self.fc1 = nn.LazyLinear(out_features=self.l2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.l2, self.l3)
        self.fc3 = nn.Linear(self.l3, 5)
        # Since we are doing multi-label classification, we use sigmoid as the
        # activation function for the last layer.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass functions of the neural network.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    # Load the data
    X_train = pd.read_csv('data/day1/train.csv', index_col=0)
    y_train = pd.read_csv('data/day1/train_labels.csv', index_col=0)
    X_val = pd.read_csv('data/day1/val.csv', index_col=0)
    y_val = pd.read_csv('data/day1/val_labels.csv', index_col=0)

    # Perform PCA
    pca = PCA(n_components='mle')
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_val = pd.DataFrame(pca.transform(X_val))

    # Get the data ready for training
    stock_dataset_train = StockDataset(X_train, y_train)
    stock_dataset_val = StockDataset(X_val, y_val)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        stock_dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        stock_dataset_val,
        batch_size=batch_size,
        shuffle=False,
    )

    # Define the model
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    model = StockLinearNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    n_epochs = 50
    train_losses_all = []
    train_counter_all = []
    test_losses = test(network=model, test_loader=val_loader)
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter = train(
            epoch=epoch,
            train_loader=train_loader,
            batch_size_train=batch_size,
            network=model,
            optimizer=optimizer,
            log_interval=10,
            save_dir='results/',
        )
        test_losses += test(network=model, test_loader=val_loader)
        train_losses_all += train_losses
        train_counter_all += train_counter

    print('Finished Training')

    fig = px.line(
        x=train_counter_all, y=train_losses_all, title='Training Loss and Test Loss'
    )
    fig.add_trace(
        px.scatter(x=test_counter, y=test_losses, color_discrete_sequence=['red']).data[
            0
        ]
    )
    fig.update_layout(
        xaxis_title='Number of training examples seen',
        yaxis_title='Loss',
    )
    fig.write_html('results/stock_linearnn.html')
