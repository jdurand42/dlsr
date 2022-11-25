import pandas as pd
import matplotlib.pyplot as plt
import sys

def histogram(src = './datasets/dataset_train.csv'):
    df = pd.read_csv(src, index_col= 'Index')
    gb = df.groupby('Hogwarts House')    
    s =[gb.get_group(x) for x in gb.groups]
    cours = df.columns[5:]  
    fig, axs = plt.subplots(3 , len(cours) // 3 + 1)
    labels = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for i, val in enumerate(cours):
        d = [x[val]  for x in s]
        axs[i%3][i // 3].hist(d, 20, stacked=True)
        axs[i%3][i // 3].set_title(val)
    fig.legend(labels) 
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('takes only one arguemnt')
        exit()
    histogram(str(sys.argv[1]))