import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression

from logreg_predict import Predictor, DataParserTest

import numpy as np
import pandas as pd
import sys
import pickle
from logistic_regression import LogisticRegression
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

export_path = "./models/models"
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
eval_data_path="data/eval/test.csv"

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
        data_path = eval_data_path
    try:
        model_path = sys.argv[2]
    except:
        model_path = export_path
    return data_path, model_path

def print_feature_importance(models, features):
    houses = list(models['houses'].keys())
    t = np.absolute(models['houses'][houses[0]]['thetas'])
    for i in range(1, len(houses)):
        t = np.append(t, np.absolute(models['houses'][houses[i]]['thetas']), axis=1)
    print(t)
    t = t[1:].transpose()
    print(t)
    fig, ax = plt.subplots()
    X = np.arange(0, len(features))
    width = 0.2
    b1 = ax.bar(X - 0.5 * width, t[0] / t[0].sum(), color="b", width=width)
    b2 = ax.bar(X - width * 1.5, t[1] / t[1].sum(), color="g", width=width)
    b3 = ax.bar(X + width * 0.5 , t[2] / t[2].sum(), color="red", width=width)
    b4 = ax.bar(X + width * 1.5, t[3] / t[3].sum(), color="grey", width=width)
    ax.set_title('Importance by feature and house')
    ax.set_ylabel("Importance thumb coefficient")
    ax.set_xticks(X, features)
    ax.legend(labels=houses)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    data_path, model_path = parse_args()

    models = load_models(export_path=model_path)
    # print(len(models))
    print(models)

    datas = DataParserTest(data_path=data_path, features=models['features'], \
        target=models['target'], normalization=models['normalization'])
    print(datas.df.head())
    print(datas.df.shape)
    print(datas.df.head())
    print(datas.X[0:5])
    print(datas.X.shape)

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
    print(final_df.head(20))

    print_feature_importance(models, features)

    # final_df.to_csv(predictions_path)
