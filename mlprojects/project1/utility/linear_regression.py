# Copyright © 2023 "Bronte" Sihan Li
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def do_single_linear_regression(
    train_data_file: str,
    test_data_file: str,
    target: str,
    features: list,
):

    train_data = pd.read_csv(train_data_file)
    test_data = pd.read_csv(test_data_file)

    fig, axs = plt.subplots(
        figsize=(13, 5), layout='constrained', ncols=len(features), nrows=1, sharey=True
    )

    # Plot each feature against target
    for feature in features:
        X_train = train_data[feature].values.reshape(-1, 1)
        y_train = train_data[target]
        X_test = test_data[feature].values.reshape(-1, 1)
        y_test = test_data[target]

        lr = LinearRegression().fit(X_train, y_train)

        print(
            f'''
        Feature: {feature}
        Training score: {lr.score(X_train, y_train)}
        Test score: {lr.score(X_test, y_test)}
        Coefficient: {lr.coef_}
        Y_intercept: {lr.intercept_}
        '''
        )

        m = lr.coef_
        b = lr.intercept_
        ax = axs[features.index(feature)]
        ax.plot(
            X_train,
            y_train,
            'o',
            c=np.random.rand(
                3,
            ),
        )
        ax.set_xlabel(feature)
        ax.plot(X_train, m * X_train + b)

    axs[0].set_ylabel(target)
    fig.show()


def do_multi_linear_regression(
    train_data_file: str,
    test_data_file: str,
    target: str,
    features: list,
):

    # Load data
    train_data = pd.read_csv(train_data_file)
    test_data = pd.read_csv(test_data_file)

    X_train = train_data[features].to_numpy(copy=True)
    y_train = train_data[target].to_numpy(copy=True)

    X_test = test_data[features].to_numpy(copy=True)
    y_test = test_data[target].to_numpy(copy=True)

    # Fit model
    lr = LinearRegression().fit(X_train, y_train)

    # Print results
    print_report(lr, X_train, y_train, X_test, y_test, features)


def do_polynomial_regression(X_train, y_train, X_test, y_test, degree: int):
    def transform_x(X, degree):
        features = ['X^1']
        # Transform features to include polynomial features, X is a vector
        t = np.copy(X)
        X_poly = np.copy(X)
        for d in range(2, degree + 1):
            # Add polynomial features as new columns
            X_poly = np.column_stack((X_poly, t**d))
            features.append(f'X^{d}')
        return X_poly, features

    X_train_poly, features = transform_x(X_train, degree)
    X_test_poly, _ = transform_x(X_test, degree)
    lr = LinearRegression().fit(X_train_poly, y_train)
    print_report(lr, X_train_poly, y_train, X_test_poly, y_test, features)
    return lr.intercept_, lr.coef_


def print_report(
    lr: LinearRegression, X_train, y_train, X_test, y_test, features: list
):
    # Print results
    print(f'Training score: {lr.score(X_train, y_train)}')
    print(f'Test score: {lr.score(X_test, y_test)}')
    print(f'Y_intercept: {lr.intercept_}')
    for feature in features:
        print(f'Feature: {feature}')
        print(f'Coefficient: {lr.coef_[features.index(feature)]}')
