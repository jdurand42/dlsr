import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression
from datetime import datetime
import os

import argparse

parser = argparse.ArgumentParser()

# now = datetime.now()
# date_time = now.strftime("%m_%d_%Y_%H:%M:%S")

parser.add_argument('file', type=str, default='data/dataset_test.csv',
                    help='path to csv file containing data')
parser.add_argument('models', type=str, default = "models/models.pkl",
                    help='path of file containing model.pkl')

predictions_folder = "./predictions/"
predictions_path = f"{predictions_folder}/houses.csv"
os.makedirs(predictions_folder, exist_ok=True)


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
        self.X = self.df[features].to_numpy()

    def zscore_(self, x, std, mean):
	    x_prime = (x - mean) / std
	    return x_prime


def load_models(export_path):
    try:
        with open(f"{export_path}", "rb") as f:
            ones = pickle.load(f)
        return ones
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":

    args = parser.parse_args()
    data_path = args.file
    models = load_models(export_path=args.models)
    predictions_path = predictions_path
    # print(len(models))
    # print(models)

    datas = DataParserTest(data_path=data_path, features=models['features'], \
        target=models['target'], normalization=models['normalization'])
    print(datas.df.head(1))
    print(datas.X[0:1])

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
            if preds[key][i] > best:
                best = preds[key][i]
                best_key = key
        final_pred.append(best_key)

    final_df = pd.DataFrame({target: final_pred})
    final_df["Index"] = range(0, len(final_df))
    final_df.set_index(["Index"], inplace=True)
    print(final_df.head())
    final_df.to_csv(predictions_path, index=True)
