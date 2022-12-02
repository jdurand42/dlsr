import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression
from datetime import datetime

export_path = "./models/models"
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
predictions_path = f"./predictions/houses.csv"

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

class Predictor:
    def __init__(self, thetas, reg_params={}):
        self.thetas = thetas
        self.reg = LogisticRegression(thetas, **reg_params)

    def predict(self, X):
        return self.reg.predict_(X)

class DataParserTest:
    def __init__(self, features=features, target=target, \
                    data_path="data/dataset_test.csv", y_labels=y_labels,\
                    normalization=None):
        self.features = features
        self.target = target
        self.y_labels = y_labels

        self.df = get_df(data_path)

        self.df = self.df

        for col in self.df[features]:
            self.df[col] = self.df[col].fillna(value=self.df[col].mean())

        if normalization is not None:
            for feature in self.df[features]:
                self.df[feature] = self.zscore_(self.df[feature].to_numpy(),normalization['stds'][feature], \
                    normalization['means'][feature])
        # self.df = self.df.dropna(axis=0).reset_index(drop=True)
        self.X = self.df[features].to_numpy()

    def zscore_(self, x, std, mean):
	    x_prime = (x - mean) / std
	    return x_prime


def load_models(export_path=export_path):
    try:
        with open(f"{export_path}", "rb") as f:
            ones = pickle.load(f)
        return ones
    except Exception as e:
        print(e)
        sys.exit(1)

def parse_args():
    try:
        data_path = sys.argv[1]
    except:
        data_path = "data/dataset_test.csv"
    try:
        model_path = sys.argv[2]
    except:
        model_path = export_path
    return data_path, model_path

if __name__ == "__main__":

    data_path, model_path = parse_args()

    models = load_models(export_path=model_path)
    # print(len(models))
    # print(models)

    datas = DataParserTest(data_path=data_path, features=models['features'], \
        target=models['target'], normalization=models['normalization'])
    print(datas.df.head(1))
    # print(datas.df.shape)
    # print(datas.df.head(1))
    print(datas.X[0:1])
    # print(datas.X.shape)

    ones = {}
    preds = {}
    for key in models['houses'].keys():
        ones[key] = Predictor(thetas=models['houses'][key]['thetas'], \
        reg_params=models['houses'][key]['reg_params'])
        preds[key] = ones[key].predict(datas.X)

    final_pred = []
    for i in range(0, len(datas.X)):
        best = -1
        for key in preds.keys():
            # print(key, preds[key][i])
            if preds[key][i] > best:
                best = preds[key][i]
                best_key = key
        final_pred.append(best_key)

    final_df = pd.DataFrame({target: final_pred})
    # print(final_df.keys())
    final_df["Index"] = range(0, len(final_df))
    final_df.set_index(["Index"], inplace=True)
    print(final_df.head())
    final_df.to_csv(predictions_path, index=True, header=True)
