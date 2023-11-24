from typing import Callable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Callable
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

root = "./datasets"

def show_matrix(predict: Callable, x, y, classes):
    y_pred = predict(x)
    score = accuracy_score(y, y_pred)
    mat = confusion_matrix(y, y_pred)
    plot_confusion_matrix(conf_mat=mat, class_names=classes,
                        show_normed=True, figsize=(7, 7))
    plt.title(f"{predict.__self__.__class__.__name__} score: {score*100:.2f}%")
    

def get_path(*path):
    return os.path.join(root, *path)