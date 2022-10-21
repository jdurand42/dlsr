import numpy as np
import pandas as pd
import sys
from logistic_regression import LogisticRegression
from sklearn.metrics import r2_score

def get_df(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(e)
        sys.exit(1)

features = ['Arithmancy', 'Astronomy', 'Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Care of Magical Creatures', 'Charms', 'Flying']

target = 'Hogwarts House'

def get_df_features(df):
    b = df[features]
    return b

def label_one_vs_all(y, target, value):
    y = y.copy()
    for i in range(0, len(y)):
        if y[i] == value:
            y[i] = 1
        else:
            y[i] = 0
    print(y.head())
    return y

def get_reg(X, Y):
    thetas = np.zeros((X.shape[1] + 1, 1))
    reg = LogisticRegression(thetas, alpha=0.000000001, max_iter=500000000)
    reg.fit_(X, Y)
    # print("thetas: ", reg.thetas)
    return reg

def perform_one_reg(X_train, Y_train, X_test, Y_test):
    # Y = binarise_y(Y_train, target_house)
    reg = get_reg(X_train, Y_train)
    y_pred = reg.predict_(X_train)
    # print(y_pred.shape)
    print("Pred", np.unique(y_pred))
    # print(Y_test.shape)
    print(f"loss: {reg.loss_(Y_train, y_pred)}")
    print(f"Score: {reg.score_(Y_train, y_pred)}")
    print(f"Sklearn score: {r2_score(Y_train, y_pred)}")
    return reg

def get_x(df, features, target):
    X = df[features].to_numpy()
    print("ici: ", X.shape)
    return X

if __name__=="__main__":

    one = "Slytherin"

    df = get_df("data/dataset_train.csv")
    df = df.dropna(axis=0).reset_index(drop=True)
    df_test = get_df("data/dataset_test.csv")
    # df_test = df_test.dropna(axis=0).reset_index(drop=True)
    # print(df.head(5))
    # print(df.columns)
    
    X_train = get_x(df, features, target)
    # print(np.unique(X_train))
    Y_train = df[target]
    print(np.unique(Y_train))
    X_test = get_x(df_test, features, target)
    Y_test = df_test[target]

    print(np.unique(X_test))

    print(X_test.shape)
    Y_train = label_one_vs_all(Y_train, 'Hogwarts House', one)
    Y_train = Y_train.to_numpy()
    Y_train = Y_train.reshape((Y_train.shape[0], 1))

    Y_test = label_one_vs_all(Y_test, 'Hogwarts House', one)
    Y_test = Y_test.to_numpy()
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    # print(Y_train)
    print("icicicicic:", X_test.shape)
    reg = perform_one_reg(X_train, Y_train, X_test, Y_test)