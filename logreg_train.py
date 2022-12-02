import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression
from TinyStatistician import TinyStatistician as Ts
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('file', type=str, default='data/dataset_train.csv',
                    help='path to csv file containing data')
parser.add_argument('-e','--early_stopping', type=int, default = None,
                    help='number of epochs needed for early stopping')
parser.add_argument('-p','--prescision', type=int, default = 5,
                    help='prescision for early stopping')
parser.add_argument('--export_path', type=str, default = "models/models.pkl",
                    help='Output path for model pkl')
 
def get_df(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(e)
        sys.exit(1)


features = ['Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration',
       'Charms', 'Flying']

target = 'Hogwarts House'

y_labels=["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

export_path = "./models/"
eval_data_path = "data/eval/test.csv"

class DataParser:
    def __init__(self, data_train_path="data/dataset_train.csv", features=features, target=target, \
                    test_split=False, ratio=None, y_labels=y_labels, normalize=False):
        self.features = features
        self.target = target
        self.y_labels = y_labels
        self.test_split = test_split
        self.ratio = ratio
        self.normalized = normalize
        self.stds = {}
        self.means = {}
        self.ts = Ts()

        self.df = get_df(data_train_path)

        self.df = self.df[[target]+features]

        for col in self.df[features]:
            self.df[col] = self.df[col].fillna(value=self.df[col].mean())

        self.get_stds_means()

        if test_split == False:
            self.df_train = self.df
            self.df_test = self.df
        else:
            self.split_df(ratio=ratio)

        self.df_test[self.features+[self.target]].to_csv(eval_data_path)

        if normalize == True:
            self.normalize()

        self.X_train = self.df_train[features].to_numpy()
        self.X_test = self.df_test[features].to_numpy()

        self.df_Y_train = self.df_train[self.target]
        self.df_Y_test = self.df_test[self.target]

        self.Ys_train = self.get_Ys(self.df_Y_train)
        self.Ys_test = self.get_Ys(self.df_Y_test)


    def get_stds_means(self):
        for feature in self.features:
            self.means[feature] = self.ts.mean(self.df[feature].to_numpy())
            self.stds[feature] = self.ts.std(self.df[feature].to_numpy())

    def normalize(self):
        for feature in self.features:
            self.df_test[feature] = self.zscore_(self.df_test[feature], self.stds[feature], self.means[feature])
            self.df_train[feature] = self.zscore_(self.df_train[feature], self.stds[feature], self.means[feature])

    def zscore_(self, x, std, mean):
        x_prime = (x - mean) / std
        return x_prime

    def _split_(self, ratio):
        train = []
        for i in self.df[self.target].unique():
            train.extend(self.df[self.df[self.target] == i].sample(frac=ratio).index)
        return train

    def split_df(self, ratio):
        if ratio == None:
            ratio = 0.8
        index = self._split_(ratio)
        split = self.df.loc[index]
        rest = self.df.drop(index)
        self.df_train= split.reset_index()
        self.df_test = rest.reset_index()


    def get_Ys(self, Y_ori):
        b = {}
        for i in range(0, len(self.y_labels)):
            b[y_labels[i]] = self.get_Y(Y_ori, y_labels[i])
        return b

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
    def __init__(self, datas, y_label, alpha=0.00001, max_iter=10000, \
                thetas=None, stochastic=False):
        self.datas = datas
        self.reg_params = {'alpha': alpha, 'max_iter': max_iter, 'stochastic': stochastic}
        self.y_label = y_label

        self.Y_train = self.datas.Ys_train[self.y_label]
        self.Y_test = self.datas.Ys_test[self.y_label]
        thetas = thetas

        if thetas is None:
            thetas = np.zeros((self.datas.X_train.shape[1] + 1, 1))
        self.reg = LogisticRegression(thetas, alpha=alpha, max_iter=max_iter, stochastic=stochastic)

    def train_reg(self, early_stopping = None, prescision = 5):
        self.reg.fit_(self.datas.X_train, self.Y_train, early_stopping = early_stopping, prescision =prescision)
        return self.reg

    def get_pred(self, X=None):
        if X is None:
            X = self.datas.X_test
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
    return X

def export_models(ones, export_path):
    with open(export_path, "wb") as f:
        pickle.dump(ones, f)

def print_data_infos(datas):
    print(datas.df.head(2))

def print_test_split_infos(datas, y_labels=y_labels, target=target):
    print("Split ratio: ", datas.ratio)
    print("train: Size: ", len(datas.df_train))
    for i in range(0, len(y_labels)):
        print(f"{y_labels[i]}: {len(datas.df_train[datas.df_train[target] == y_labels[i]])}")
    print("test: Size: ", len(datas.df_test))
    for i in range(0, len(y_labels)):
        print(f"{y_labels[i]}: {len(datas.df_test[datas.df_test[target] == y_labels[i]])}")

if __name__=="__main__":
    args = parser.parse_args()
    print(args.file)
    datas = DataParser(data_train_path=args.file, \
                        test_split=True, \
                        ratio=0.8, \
                        normalize=True,
                        )
    print_data_infos(datas)

    y_labels=y_labels

    print_test_split_infos(datas, y_labels, target)
    preds = {}
    ones = {}
    ones['houses'] = {}
    models = {}
    models['houses'] = {}
    if datas.normalized == True:
        models['normalization'] = {'means': datas.means, 'stds': datas.stds}
    models['features'] = features
    models['target'] = target
    fig, axs = plt.subplots(2)
   
    for i in range(0,len(y_labels)):
        print("--------------")
        print("")
        print(y_labels[i])
        one = OneVersusAll(datas, y_labels[i], \
                            max_iter=3000, alpha=0.1, stochastic=False)
        one.train_reg(args.early_stopping, args.prescision)
        metrics = one.evaluate()
        print(metrics)
        ones['houses'][y_labels[i]] = one
        preds[y_labels[i]] = one.get_pred()
        models['houses'][y_labels[i]] = {
            'thetas': one.reg.thetas,
            'score': metrics['score'],
            'loss': metrics['loss'],
            'reg_params': one.reg_params,
            'y_label': one.y_label,
        }
        axs[0].plot(one.reg.losses)
        axs[1].plot(one.reg.r2s)
        print("-----------")

    axs[0].set_ylabel('loss')
    axs[1].set_ylabel('r2')
    axs[1].set_xlabel('epochs')
    axs[0].legend(y_labels)

    plt.show()

    final_pred = datas.df_Y_test.copy()

    for i in range(0, len(final_pred)):
        best = -1
        for key in preds.keys():
            if preds[key][i] > best:
                best = preds[key][i]
                best_key = key
        final_pred[i] = best_key
    print(final_pred.head(2))

    b = final_pred.to_numpy()
    b2 = datas.df_Y_test.copy()

    pos = 0
    for i in range(0,len(final_pred)):
        if b[i] == b2[i]:
            pos += 1
    score = pos / len(final_pred)
    print(f"Score: {score}")

    export_models(models, args.export_path)
