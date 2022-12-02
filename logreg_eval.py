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

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('file', type=str, default='data/eval/test.csv',
                    help='path to csv file test split containing data')
parser.add_argument('models', type=str, default = "models/models.pkl",
                    help='path of file containing model.pkl')
 
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

def load_models(export_path):
    try:
        with open(export_path, "rb") as f:
            ones = pickle.load(f)
        return ones
    except Exception as e:
        print(e)
        sys.exit(1)

def print_feature_importance(models, features):
    houses = list(models['houses'].keys())
    t = np.absolute(models['houses'][houses[0]]['thetas'])
    for i in range(1, len(houses)):
        t = np.append(t, np.absolute(models['houses'][houses[i]]['thetas']), axis=1)
    # print(t)
    t = t[1:].transpose()
    # print(t)
    fig, ax = plt.subplots()
    X = np.arange(0, len(features))
    width = 0.2
    b1 = ax.bar(X - 0.5 * width, t[0] / t[0].sum(), color="b", width=width)
    b2 = ax.bar(X - width * 1.5, t[1] / t[1].sum(), color="g", width=width)
    b3 = ax.bar(X + width * 0.5 , t[2] / t[2].sum(), color="red", width=width)
    b4 = ax.bar(X + width * 1.5, t[3] / t[3].sum(), color="grey", width=width)
    ax.set_title('Importance by feature and house')
    ax.set_ylabel("Importance")
    ax.set_xticks(X, features)
    ax.legend(labels=houses)
    plt.tight_layout()
    plt.show()


def get_idx(x, label):
	for i in range(0, len(x)):
		if x[i] == label:
			return i
	return 0

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
		y:a numpy.array for the correct labels
		y_hat:a numpy.array for the predicted labels
		labels: optional, a list of labels to index the matrix.
		This may be used to reorder or select a subset of labels. (default=None)
		df_option: optional, if set to True the function will return a pandas DataFrame
		instead of a numpy array. (default=False)
	Return:
		The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
		None if any error.
	Raises:
		This function should not raise any Exception.
	"""
	uni = np.unique(np.concatenate((y_true, y_hat)))
	if labels is None:
		labels = uni
	dim = len(labels)
	mat = np.zeros((dim, dim), dtype=int)

	y = y_true
	for i in range(0, len(y)):
		if y[i][0] in labels:
			idx_label = get_idx(labels, y[i][0])
		if y[i][0] == y_hat[i][0]:
			mat[idx_label][idx_label] += 1
		elif y[i][0] != y_hat[i][0] and y_hat[i][0] in labels:
			mat[idx_label][get_idx(labels, y_hat[i][0])] += 1
	if df_option is False:
		return mat
	return pd.DataFrame(mat, columns=labels, index=labels)
	# Df options true


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.file
    models = load_models(args.models)

    datas = DataParserTest(data_path=data_path, features=models['features'], \
        target=models['target'], normalization=models['normalization'])

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
    print(final_df.head(5))

    print_feature_importance(models, features)
    Y = datas.df[target].to_numpy().reshape(datas.df[target].shape[0], 1)
    mat = confusion_matrix_(Y, final_df.to_numpy(), df_option=True)
    print(mat)
