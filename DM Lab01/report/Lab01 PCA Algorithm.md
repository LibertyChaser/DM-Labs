# Lab01 PCA Algorithm

---

[toc]

Student ID: 19011418

Author: Sonqing Zhao, Minzu University of China

Written at Nov 11^th^, 2021



> [A Step-by-Step Explanation of Principal Component Analysis (PCA)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)]
>
> [Sklearn数据预处理：scale, StandardScaler, MinMaxScaler, Normalizer](https://blog.csdn.net/u013402321/article/details/79043402)
>
> [数据归一化（Feature Scaling）](https://www.cnblogs.com/volcao/p/9089716.html)
>
> [6.3. Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-data)
>
> [Eigenvalues and Eigenvectors](https://personal.math.ubc.ca/~pwalls/math-python/linear-algebra/eigenvalues-eigenvectors/)
>
> [Principal Component Analysis from Scratch in Python](https://www.askpython.com/python/examples/principal-component-analysis)
>
> [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)
>
> [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

---

## Lab Purpose

Master the principle of PCA algorithm

## Lab Requirements

1. Understand the data dimensionality reduction process
2. Use Python or other tools to implement the PCA algorithm expertly

## Lab Equipment

1. A computer
2. Python or other tools

## Lab Procedure

### Explain the PCA dimensionality reduction process

Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

1. Standardize the range of continuous initial variables

When data sets have large differences between the ranges of initial variables, those variables with larger ranges will dominate over those with small ranges.
$$
z = \frac{\text{value} - \text{mean}}{\text{standard deviation}}
$$
Or using simply minus way still can be a good way to solve problem here.
$$
z = \text{value} - \text{mean}
$$

2. Compute the covariance matrix to identify correlations

3. Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components

4. Create a feature vector to decide which principal components to keep

5. Recast the data along the principal components axes

### Select data set

Select all 4 data set: sepal length, sepal width, petal length and petal widthCode

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
species = iris.target #species
```

Result

```python
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
 ...
       [6.5, 3. , 5.2, 2. ],
       [6.2, 3.4, 5.4, 2.3],
       [5.9, 3. , 5.1, 1.8]])
```

### Standardization

Code

```python
std_data = input_data - np.mean(input_data)
```

Result

```python
[[ 1.6355  0.0355 -2.0645 -3.2645], [ 1.4355 -0.4645 -2.0645 -3.2645], [ 1.2355 -0.2645 -2.1645 -3.2645], [ 1.1355 -0.3645 -1.9645 -3.2645], [ 1.5355  0.1355 -2.0645 -3.2645], [ 1.9355  0.4355 -1.7645 -3.0645], [ 1.1355 -0.0645 -2.0645 -3.1645], [ 1.5355 -0.0645 -1.9645 -3.2645], [ 0.9355 -0.5645 -2.0645 -3.2645], [ 1.4355 -0.3645 -1.9645 -3.3645], [ 1.9355  0.2355 -1.9645 -3.2645], [ 1.3355 -0.0645 -1.8645 -3.2645], [ 1.3355 -0.4645 -2.0645 -3.3645], [ 0.8355 -0.4645 -2.3645 -3.3645], [ 2.3355  0.5355 -2.2645 -3.2645], [ 2.2355  0.9355 -1.9645 -3.0645], [ 1.9355  0.4355 -2.1645 -3.0645], [ 1.6355  0.0355 -2.0645 -3.1645], [ 2.2355  0.3355 -1.7645 -3.1645], [ 1.6355  0.3355 -1.9645 -3.1645], [ 1.9355 -0.0645 -1.7645 -3.2645], [ 1.6355  0.2355 -1.9645 -3.0645], [ 1.1355  0.1355 -2.4645 -3.2645], [ 1.6355 -0.1645 -1.7645 -2.9645], [ 1.3355 -0.0645 -1.5645 -3.2645], [ 1.5355 -0.4645 -1.8645 -3.2645], [ 1.5355 -0.0645 -1.8645 -3.0645], [ 1.7355  0.0355 -1.9645 -3.2645], [ 1.7355 -0.0645 -2...
```

### Covariance matrix

Code

```python
cov_mat = np.cov(std_data)
```

Result

```python
[[ 0.68569351 -0.042434    1.27431544  0.51627069], [-0.042434    0.18997942 -0.32965638 -0.12163937], [ 1.27431544 -0.32965638  3.11627785  1.2956094 ], [ 0.51627069 -0.12163937  1.2956094   0.58100626]]
```

### Eigenvalues and eigenvectors

Code

```python
eig_val, eig_vec = np.linalg.eig(cov_mat)
```

Result

`eig_val`

```python
[4.22824171 0.24267075 0.0782095  0.02383509]
```

`eig_vec`

```python
[[ 0.36138659 -0.65658877 -0.58202985  0.31548719], [-0.08452251 -0.73016143  0.59791083 -0.3197231 ], [ 0.85667061  0.17337266  0.07623608 -0.47983899], [ 0.3582892   0.07548102  0.54583143  0.75365743]]
```

### Reduce data to k dimensions

Reduce the data to k dimensions and sort it by the size of the eigenvalues, select the eigenvectors corresponding to the first k eigenvalues.

The value of k dimensions of k can be selected based on the original data set, if the iris data set is used, k can take the value 2.

Code

```python
# sort the eigenvalues in descending order
sorted_index = np.argsort(eig_val)[::-1]

sorted_eig_val = eig_val[sorted_index]
# similarly sort the eigenvectors
sorted_eig_vec = eig_vec[:, sorted_index]

# select the first n eigenvectors, n is desired dimension
# of our final reduced data.
n_components = 2  # you can select any number of components.
eig_vec_subset = sorted_eig_vec[:, 0:n_components]
```

Result

```python
          PC1       PC2  target
0   -2.350184 -1.704107       0
1   -2.380200 -1.207709       0
2   -2.555049 -1.239761       0
3   -2.411402 -1.066411       0
4   -2.394775 -1.711465       0
..        ...       ...     ...
145  2.278051 -1.572242       2
146  1.861108 -1.009393       2
147  2.098287 -1.463569       2
148  2.234883 -1.501338       2
149  1.724130 -1.102049       2
```

## Lab Result

Test code

```python
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import PCA

if __name__ == '__main__':
    iris = datasets.load_iris()
    target = iris.target
    data = PCA.PCA()
    # Creating a Pandas DataFrame of reduced Dataset
    principal_df = pd.DataFrame(data, columns=['PC1', 'PC2'])
    # Concat it with target variable to create a complete Dataset
    principal_df = pd.concat([principal_df, pd.DataFrame(target)], axis=1)
    principal_df = principal_df.rename(columns={principal_df.columns[2]: 'target'}, inplace=False)
    print(principal_df)

    plt.figure()
    plt.title("Iris Dataset PCA", size=14)
    plt.scatter(principal_df["PC1"], principal_df["PC2"], c=principal_df["target"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

```

Result plot using data preprocessed by simply minus way

<img src="Lab01 PCA Algorithm.assets/PCA1.png" alt="pca1" style="zoom:50%;" />

Result plot using data divided during the step of preprocessing.

<img src="Lab01 PCA Algorithm.assets/PCA2.png" alt="pca1" style="zoom:50%;" />

## Improvement and innovation

 Comparing different data preprocessing methods.

Still don’t know why complex way to preproess data can’t get expected answer.