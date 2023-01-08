import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

def format_time(elapsed):
    """Converts timestamp to readable format"""
    return time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed))

def save_pickle(fname:Union[str, Path], object):
    """Stores an object in pickle format

    Parameters
    ----------
    fname : Union[str, Path]
        export full path (with filename)
    object 
        It can be any object
    """

    if isinstance(fname, str):
        fname = Path(fname)

    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(object, f)

def load_pickle(fname:Union[str, Path]):
    """Load pickle file

    Parameters
    ----------
    fname : Union[str, Path]
        Full file path

    Returns
    -------
        The loaded object
    """

    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, norm:bool, fullpath:Union[str, Path], figsize=(15,10)):
    """Plot and store as image the confusion matrix

    Parameters
    ----------
    y_true : np.ndarray
        Groundtruth
    y_pred : np.ndarray
        Prediction
    norm : bool
        Normalize or not the confusion matrix
    fullpath : Union[str, Path]
        Export full path
    figsize : tuple, optional
        Image size, by default (15,10)
    """

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