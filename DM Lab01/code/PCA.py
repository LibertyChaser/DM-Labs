from sklearn import datasets
import numpy as np
from sklearn import preprocessing


def PCA(input_data=None, num_components=None):
    if input_data is None:
        input_data = datasets.load_iris().data
    if num_components is None:
        num_components = 2

    std_data = preprocessing.scale(input_data)
    # std_data = input_data - np.mean(input_data)
    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(std_data, rowvar=False)

    # Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eig_val, eig_vec = np.linalg.eig(cov_mat)

    # sort the eigenvalues in descending order
    sorted_index = np.argsort(eig_val)[::-1]

    # sorted_eig_val = eig_val[sorted_index]
    # similarly sort the eigenvectors
    sorted_eig_vec = eig_vec[:, sorted_index]

    # select the first n eigenvectors, n is desired dimension
    # of our final reduced data.

    eig_vec_subset = sorted_eig_vec[:, 0:num_components]
    # print(eig_vec_subset)

    # Transform the data
    reduced_mat = np.dot(eig_vec_subset.transpose(), std_data.transpose()).transpose()
    return reduced_mat

