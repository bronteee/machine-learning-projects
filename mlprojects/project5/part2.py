import torch
from utils.train_test import train, test
from utils.dataset import StockDataset
from utils.net import StockLinearNN
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from part3_4 import test_accuracy


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

    print(model)

    # Train model
    n_epochs = 10
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
    test_acc = test_accuracy(model)
    print(f'Test accuracy: {test_acc}')

    fig = px.line(
        x=train_counter_all,
        y=train_losses_all,
        title=f'Training Loss and Test Loss, test accuracy: {test_acc}',
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
