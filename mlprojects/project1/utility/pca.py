# Copyright © 2023 "Bronte" Sihan Li
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from textwrap import dedent
import numpy as np
import pandas as pd


class PCA:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.mean = None
        self.std = None
        self._eigenvalues = None
        self._eigenvectors = None

    def pca(self, data, normalize=True):
        """
        computes the principal components of the given data
        returns the means, standard deviations, eigenvalues, eigenvectors, and projected data
        """
        # assign to A the data as a numpy matrix
        A = data.to_numpy(copy=True)
        # assign to m the mean values of the columns of A
        m = np.mean(A, axis=0)
        # assign to D the difference matrix A - m
        D = A - m
        # if normalize is true
        if normalize:
            #    Compute the standard deviations of each column
            std = np.std(A, axis=0)
        # else
        else:
            #    Assign all 1s to the standard deviation vector (1 for each column)
            std = np.ones(A.shape[1])
        # Divide each column by its standard deviation vector
        #    (hint: this can be done as a single operation)
        D = D / std
        # assign to U, S, V the result of running np.svd on D, with full_matrices=False
        U, S, V = np.linalg.svd(D, full_matrices=False)
        # the eigenvalues of cov(A) are the squares of the singular values (S matrix)
        #   divided by the degrees of freedom (N-1). The values are sorted.
        eigenvalues = (S**2) / (A.shape[0] - 1)
        # project the data onto the eigenvectors. Treat V as a transformation
        #   matrix and right-multiply it by D transpose. The eigenvectors of A
        #   are the rows of V. The eigenvectors match the order of the eigenvalues.
        projected = np.dot(D, V.T)
        # create a new data frame out of the projected data
        # return the means, standard deviations, eigenvalues, eigenvectors, and projected data

        self.mean = m
        self.std = std
        self._eigenvalues = eigenvalues
        self._eigenvectors = V

        return m, std, eigenvalues, V, pd.DataFrame(projected)

    def do_pca(self, normalize=True) -> pd.DataFrame:
        mean, std, eigenvalues, eigenvectors, projected = self.pca(
            self.data, normalize=normalize
        )
        print(
            dedent(
                f"""Mean: {mean}
Standard Deviation: {std}
Eigenvalues: {eigenvalues}
Eigenvectors: \n{eigenvectors}"""
            )
        )
        return projected

    def transform(self, X: np.ndarray, n_components: int = None) -> np.ndarray:
        assert X.shape[1] == self._eigenvectors.shape[0]

        if n_components is None:
            n_components = self._eigenvectors.shape[0]

        D = X - self.mean
        D = D / self.std

        return np.dot(D, self._eigenvectors.T)[:, :n_components]
    
    @property
    def components(self):
        return self._eigenvectors
