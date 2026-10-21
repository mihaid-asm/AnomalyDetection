import pyod
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score
import numpy as np

X_train, y_train = (
    pyod.utils.generate_data(1000, 0, 1, 0.1, train_only=True))

mu = np.mean(X_train)
sigma = np.std(X_train)

X_train = np.abs((X_train - mu) / sigma)
z_threshold = np.quantile(X_train, 0.9)
y_pred = (X_train > z_threshold).astype(int)

matrix = confusion_matrix(y_train, y_pred)
print(matrix)
tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
print(f"TNR = {tnr}")
print(f"TPR = {tpr}")
print(f"BA = {(tnr + tpr) / 2}")