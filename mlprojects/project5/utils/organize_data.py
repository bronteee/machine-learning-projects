# Copyright (c) 2023 Bronte Sihan Li
# License: MIT License

import pandas as pd
from sklearn.impute import SimpleImputer
import re
import datetime

if __name__ == '__main__':

    DATA_PATHS = [
        'data/CNNpred/Processed_DJI.csv',
        'data/CNNpred/Processed_NASDAQ.csv',
        'data/CNNpred/Processed_NYSE.csv',
        'data/CNNpred/Processed_RUSSELL.csv',
        'data/CNNpred/Processed_S&P.csv',
    ]
    stock_dfs = []
    labels = []

    for f in DATA_PATHS:
        stock_index = re.match(r'.*Processed_(.*).csv', f).group(1)
        # read data
        df = pd.read_csv(f)
        dates = pd.to_datetime(df['Date'])
        df.drop(columns=['Date', 'Name'], inplace=True)
        # Add column for label
        # label data as 0 or 1 for down or up, use data from previous day
        label_df = pd.DataFrame()
        label_df[f'{stock_index}_label'] = [
            0 if (j - i < 0) else 1 for i, j in zip(df['Close'], df['Close'].shift(1))
        ]
        # fill in missing data with average of previous and next day
        imp = SimpleImputer(missing_values=pd.NA, strategy='median')
        transformed = imp.fit_transform(df)
        transformed_df = pd.DataFrame(transformed, columns=df.columns)
        stock_dfs.append(transformed_df)
        labels.append(label_df)

    # combine all data into one dataframe
    df = pd.concat(stock_dfs, axis=1)

    # combine all labels into one dataframe
    label_df = pd.concat(labels, axis=1)
    df = pd.concat([df, label_df], axis=1)
    # Mark data for train/test
    # Use data from 2010-2016 for training, 2017 for testing
    df['train_test'] = [0 if (i < datetime.datetime(2017, 1, 1)) else 1 for i in dates]
    # save data
    df.to_csv('data/day1prediction.csv')
    # save labels
    label_df.to_csv('data/day1labels.csv')
