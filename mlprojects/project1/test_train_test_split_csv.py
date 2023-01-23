import pytest
from utility import train_test_split_csv


class TestTrainTestSplitCsv:
    def test_repeatable_split(self):
        """
        Test that the same split is produced when the same seed is used"""
        splitter1 = train_test_split_csv.TrainTestSplit(
            'data/health_expenditure.csv', 0.8, 32
        )
        train_data1, test_data1 = splitter1.train_test_split(to_csv=False)

        splitter2 = train_test_split_csv.TrainTestSplit(
            'data/health_expenditure.csv', 0.8, 32
        )
        train_data2, test_data2 = splitter2.train_test_split(to_csv=False)

        assert train_data1.equals(train_data2)
        assert test_data1.equals(test_data2)
