import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression
from datetime import datetime

export_path = "./models/models"
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
predictions_path = f"./predictions/prediction_{date_time}.csv"

def get_df(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(e)
        sys.exit(1)

features = ['Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Charms', 'Flying']

target = 'Hogwarts House'

y_labels=["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

class Predictor:
    def __init__(self, thetas, y_label=None, reg_params={}, score=None):
        self.thetas = thetas
        self.reg = LogisticRegression(thetas, **reg_params)
        self.score = score
    
    def predict(self, X):
        return self.reg.predict_(X)

class DataParserTest:
    def __init__(self, features=features, target=target, \
                    data_path="data/dataset_test.csv", y_labels=y_labels):
        self.features = features
        self.target = target
        self.y_labels = y_labels

        self.df_raw = get_df(data_path)

        self.df = self.df_raw[features]
        # self.df = self.df.dropna(axis=0).reset_index(drop=True)

        for col in self.df:
            self.df[col] = self.df[col].fillna(value=self.df[col].mean())

        self.X = self.df.to_numpy()

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
    datas = DataParserTest(data_path=data_path)
    print(datas.df_raw.head())
    print(datas.df_raw.shape)
    print(datas.df.head())
    print(datas.X[0:5])
    print(datas.X.shape)

    models = load_models(export_path=model_path)
    print(len(models))
    print(models)

    ones = {}
    preds = {}
    for key in models.keys():
        ones[key] = Predictor(thetas=models[key]['thetas'], y_label=key, reg_params=models[key]['reg_params'], \
                score=models[key]['score'])
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
    print(final_df.head())

    final_df.to_csv(predictions_path)

    print(datas.df.describe())
