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
