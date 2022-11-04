import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatter_plot(src = './datasets/dataset_train.csv'):
    df = pd.read_csv(src, index_col= 'Index')
    pd.plotting.scatter_matrix(df[ df.columns[5:] ])
    plt.show()


if __name__ == '__main__':
    scatter_plot()