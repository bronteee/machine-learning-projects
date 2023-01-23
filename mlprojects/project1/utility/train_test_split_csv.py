# Copyright © 2023 "Bronte" Sihan Li
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import pandas as pd
import random
import argparse
from typing import Optional
from datetime import datetime
from textwrap import dedent
from typing import Tuple


class TrainTestSplit:
    def __init__(
        self, data_csv: str, split_ratio: float, random_state: Optional[int] = 0
    ) -> None:
        self.data_filename = data_csv.removesuffix('.csv')
        self.data = pd.read_csv(data_csv)
        self.split_ratio = split_ratio
        self.random_state = random_state

    @property
    def train_size(self):
        return int(self.data.shape[0] * self.split_ratio)

    @property
    def test_size(self):
        return self.data.shape[0] - self.train_size

    @property
    def data_size(self):
        return self.data.shape[0]

    def train_test_split(self, to_csv=True) -> Tuple[pd.DataFrame]:

        test_filename, train_filename = self._generate_filenames()
        self.print_metadata(test_filename, train_filename)
        # create pseudorandom generator
        random.seed(self.random_state)
        # shuffle indexes for each class of data based on split
        test_indexes = [
            random.randint(0, self.data_size) for _ in range(self.test_size)
        ]

        test_df = self.data.loc[test_indexes, :]
        train_df = self.data.drop(index=test_indexes)
        if to_csv is True:
            test_df.to_csv(test_filename)
            train_df.to_csv(train_filename)
        return train_df, test_df

    def _generate_filenames(self) -> Tuple[str]:
        # Use timestamp for filenames
        time_now = str(datetime.now())
        return (
            f'{self.data_filename}_train_{time_now}.csv',
            f'{self.data_filename}_test_{time_now}.csv',
        )

    def print_metadata(self, test_filename: str, train_filename: str) -> None:
        """Print out the percent and number of samples put into the test set,
        the percent and number of samples put into the training set,
        and the names of the test and train files"""
        print(
            dedent(
                f"""
        Sample Size: {self.data_size}
        Train-test Split ratio: {self.split_ratio}
        Number of samples in training set: {self.train_size}
        Number of samples in test set: {self.test_size}
        Test set file name: {test_filename}
        Training set file name: {train_filename}
        """
            )
        )


def split_data_cmd(parser: argparse.ArgumentParser):
    parser.add_argument(
        "csvfile", type=str, help="CSV file containing dataset for splitting"
    )
    parser.add_argument("split", type=float, help="train test split ratio")
    parser.add_argument(
        "random_state",
        type=int,
        help="Seed for the random number generator",
        default=0,
    )
    args = parser.parse_args()
    print(args.csvfile)
    print(args.split)
    splitter = TrainTestSplit(args.csvfile, args.split, args.random_state)
    splitter.train_test_split()


def main():
    parser = argparse.ArgumentParser()
    split_data_cmd(parser)


if __name__ == '__main__':
    main()
