# Copyright © 2023 "Bronte" Sihan Li

import torch
import pandas as pd


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


class Stock2DDataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, transform=None, days=60):
        self.X = X
        self.y = y
        self.transform = transform
        self.days = days

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if idx < self.days:
            x = torch.from_numpy(self.X.iloc[0 : self.days].to_numpy()).float()
        else:
            x = torch.from_numpy(
                self.X.iloc[(idx - self.days) : idx].to_numpy()
            ).float()
        x = x.view(1, self.days, -1)
        y = torch.from_numpy(self.y.iloc[idx].to_numpy()).float()
        if self.transform:
            x = self.transform(x)
        return x, y
