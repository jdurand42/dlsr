import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression
# from sklearn.metrics import r2_score

def get_df(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(e)
        sys.exit(1)

# A faire dans unconf.ini

features = ['Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Charms', 'Flying']

target = 'Hogwarts House'

y_labels=["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

export_path = "./models/"

class DataParser:
    def __init__(self, data_train_path="data/dataset_train.csv", features=features, target=target, \
                    test_split=False, y_labels=y_labels):
        self.features = features
        self.target = target
        self.y_labels = y_labels


        self.df_train, self.df_train_cleaned = self.parse_dfs(data_train_path)

        if test_split == False:
            self.df_test = self.df_train
            self.df_test_cleaned = self.df_train_cleaned

        else:
            # train test split
            pass
        
        self.X_train = self.df_train_cleaned[features].to_numpy()
        self.X_test = self.df_test_cleaned[features].to_numpy()

        self.df_Y_train = self.df_train_cleaned[self.target]
        self.df_Y_test = self.df_test_cleaned[self.target]

        # self.Y_train = self.df_train_cleaned[target]
        # self.Y_test = self.df_train_cleaned[target]

        self.Ys_train = self.get_Ys(self.df_Y_train)
        self.Ys_test = self.get_Ys(self.df_Y_test)

    def get_Ys(self, Y_ori):
        b = {}
        for i in range(0, len(self.y_labels)):
            b[y_labels[i]] = self.get_Y(Y_ori, y_labels[i])
        return b

    def parse_dfs(self, path):
        df_raw = get_df(path)
        df_cleaned = df_raw.dropna(axis=0).reset_index(drop=True)
        return df_raw, df_cleaned

    def label_one_vs_all(self, y, value):
        y = y.copy()
        for i in range(0, len(y)):
            if y[i] == value:
                y[i] = 1
            else:
                y[i] = 0
        return y

    def get_Y(self, Y_ori, label):
        Y = self.label_one_vs_all(Y_ori, label)
        Y = Y.to_numpy()
        Y = Y.reshape((Y_ori.shape[0], 1))
        return Y

    def get_data_to_predict():
        pass


class OneVersusAll:
    def __init__(self, datas, y_label, reg_params={'alpha': 0.00001, 'max_iter': 10000}, \
                thetas=None):
        self.datas = datas
        self.reg_params = reg_params
        self.y_label = y_label

        self.Y_train = self.datas.Ys_train[self.y_label]
        self.Y_test = self.datas.Ys_test[self.y_label]
        thetas = thetas

        if thetas is None:
            thetas = np.zeros((self.datas.X_train.shape[1] + 1, 1))
        self.reg = LogisticRegression(thetas, **reg_params)
    
    def train_reg(self):
        self.reg.fit_(self.datas.X_train, self.Y_train)
        # print(self.reg.thetas)
        return self.reg
    
    def get_pred(self, X=None):
        if X is None:
            X = self.datas.X_train
        y_pred = self.reg.predict_(X)
        return y_pred
    
    def evaluate(self, X=None,Y=None):
        if X is None:
            X = self.datas.X_test
        if Y is None:
            Y = self.Y_test
        y_pred = self.get_pred(X)
        
        metrics = {'loss': self.reg.loss_(Y, y_pred),
                    'score': self.reg.score_(Y, self.binarise_y(y_pred,0.5)),
                    'r2': self.reg.r2_(Y,y_pred)
                    }
        return metrics
    
    def binarise_y(self, y, treshold):
	    y = y.copy()
	    for i in range(0, len(y)):
		    if y[i][0] >= treshold:
			    y[i][0] = 1
		    else:
			    y[i][0] = 0
	    return y


def get_x(df, features, target):
    X = df[features].to_numpy()
    # print("ici: ", X.shape)
    return X

# class OneVsAll:
#     def __init__(self, data_path="data/dataset_train.csv", features=features, target=target, \
#                      )

def export_models(ones, export_path=export_path):
    with open(f"{export_path}/models", "wb") as f:
        pickle.dump(ones, f)


if __name__=="__main__":
    datas = DataParser()
    print(datas.df_train.head())
    print(datas.df_train_cleaned.head())
    print(datas.df_Y_train.head())
    print(datas.X_train[0:5])
    print(datas.X_train.shape)
    for key in datas.Ys_train.keys():
        print(key, datas.Ys_train[key][0:5])
        print(datas.Ys_train[key].shape)
    
    y_labels=y_labels

    preds = {}
    ones = {}
    models = {}
    for i in range(0,len(y_labels)):

        print("--------------")
        print("")
        print(y_labels[i])
        one = OneVersusAll(datas, y_labels[i])
        one.train_reg()
        metrics = one.evaluate()
        print(metrics)
        ones[y_labels[i]] = one
        preds[y_labels[i]] = one.get_pred()
        models[y_labels[i]] = {
            'thetas': one.reg.thetas,
            'score': metrics['score'],
            'reg_params': one.reg_params,
            'y_label': one.y_label,
        }
        print("-----------")
    print(preds)
    
    final_pred = datas.df_Y_test.copy()
    for i in range(0, len(final_pred)):
        best = -1
        for key in preds.keys():
            if preds[key][i] > best:
                # print(i)
                best = preds[key][i]
                best_key = key
        # print(i, best_key)
        final_pred[i] = best_key
    # best_key
    print(final_pred.head())

    b = final_pred.to_numpy()
    print(b)
    b2 = datas.df_train_cleaned[target].to_numpy()
    print(b.shape, b2.shape)

    pos = 0
    for i in range(0,len(final_pred)):
        if b[i] == b2[i]:
            pos += 1
    score = pos / len(final_pred)
    print(f"Score: {score}")

    export_models(models)



    # one = "Ravenclaw"

    # df = get_df("data/dataset_train.csv")
    # df = df.dropna(axis=0).reset_index(drop=True)
    # df_test = get_df("data/dataset_test.csv")
    # # df_test = df_test.dropna(axis=0).reset_index(drop=True)
    # # print(df.head(5))
    # # print(df.columns)
    
    # X_train = get_x(df, features, target)
    # # print(np.unique(X_train))
    # Y_train = df[target]
    # print(np.unique(Y_train))
    # X_test = get_x(df_test, features, target)
    # Y_test = df_test[target]

    # print(np.unique(X_test))

    # print(X_test.shape)
    # Y_train = label_one_vs_all(Y_train, 'Hogwarts House', one)
    # Y_train = Y_train.to_numpy()
    # Y_train = Y_train.reshape((Y_train.shape[0], 1))

    # Y_test = label_one_vs_all(Y_test, 'Hogwarts House', one)
    # Y_test = Y_test.to_numpy()
    # Y_test = Y_test.reshape((Y_test.shape[0], 1))
    # # print(Y_train)
    # print("icicicicic:", X_test.shape)
    # reg = perform_one_reg(X_train, Y_train, X_test, Y_test)