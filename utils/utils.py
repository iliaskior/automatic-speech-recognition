import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

def format_time(elapsed):
    return time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed))

def save_pickle(fname, object):
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(object, f)

def load_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_confusion_matrix(y_true, y_pred, norm, fullpath, figsize=(15,10)):

    if not isinstance(fullpath, Path):
        fullpath = Path(fullpath)

    labels = np.unique(y_true).tolist()
    if norm:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        fmt = '.2'
    else:
        cm = confusion_matrix(y_true, y_pred)   
        fmt = 'd'

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    acc = round(accuracy_score(y_true, y_pred), 2)

    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=True, cmap='Greens', cbar = False, fmt = fmt, annot_kws={"size": 35 / np.sqrt(len(cm_df))})
    plt.ylabel('Actal Values')
    plt.yticks(rotation = 0)
    plt.xlabel('Predicted Values')
    plt.title(f'Accuracy: {acc}', fontsize=15)
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fullpath), transparent = True, bbox_inches='tight')
    plt.show(block=False)