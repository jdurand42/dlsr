from math import sqrt
import pandas as pd
import sys

def var(x, mean, m):
    sum = 0
    for i in range(0, m):
        sum += ((x[i] - mean) * (x[i] - mean))
    return float(sum / m)

def f_std(x, mean, l):
    return float(sqrt(var(x, mean, l)))

def percentile(a, p):
    if p == 1:
        return a[-1]
    n = p *(len(a) - 1)
    return a[int(n)] + (n - int(n)) * (a[int(n) + 1] - a[int(n)])

def all_sort(x, l):
    x.sort()
    
    if l % 2 != 0:
        median = float(x[len(x) // 2])
    else:
        median = float((x[len(x) // 2] + x[len(x) // 2 - 1]) / 2)
    return percentile(x, 0.25), median ,percentile(x, 0.75), x[0], x[-1]

def all_sum(x):
    l = len(x)
    sum = 0
    for i in range(0, l):
        sum += x[i]
    mean = float(sum / l)

    return sum, mean, l

def df_to_list(df, column):
    ls = df[column].values.tolist()
    cleaned_ls = [x for x in ls if str(x) != 'nan']
    return cleaned_ls, len(ls)


def describe_one(x, l):
    sum, mean, count = all_sum(x)
    p25 , p50, p75, min, max = all_sort(x, count)
    std = f_std(x, mean, count)
    return [count, mean, std, min, p25, p50, p75, max, sum, int(l - count), max - min, std * std, 100 * std / mean]


def describe(src = 'data/dataset_train.csv'):
    pd.set_option("display.max_rows", None, "display.max_columns", None,'display.max_colwidth', -1)
    ds = pd.DataFrame()
    df = pd.read_csv(src, index_col= 'Index')
    columns = df.dtypes[(df.dtypes == 'float') | (df.dtypes == 'int')].index
    for column in columns:
        ds.insert(0, column, describe_one(*df_to_list(df, column)))
    ds.insert(0,'', ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'sum', 'Nones', 'range', 'var','coef var' ])
    ds = ds.set_index('')
    print(ds)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('takes only one arguemnt')
        exit()
    describe(str(sys.argv[1]))

