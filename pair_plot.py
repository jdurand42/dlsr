import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pair_plot(src = 'data/dataset_train.csv'):
    df = pd.read_csv(src, index_col= 'Index')
    sns.pairplot(df, hue= "Hogwarts House",vars = df.columns[5:],
                kind = 'scatter', corner=True)
    plt.show()

if __name__ == '__main__':
    pair_plot()