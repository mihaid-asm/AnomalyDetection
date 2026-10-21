import pyod
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score

X_train, X_test, y_train, y_test = (
    pyod.utils.generate_data(400, 100, 2, 0.1))

custom_cmap = ListedColormap(["cyan", "pink"])

with plt.style.context("dark_background"):
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=custom_cmap,
        alpha=0.5,
    )

    plt.colorbar(scatter, ticks=[0, 1])
    plt.show()

forest = IForest(contamination=0.2, random_state=42)
forest.fit(X_train)

y_train_pred = forest.labels_
y_train_scores = forest.decision_scores_

y_test_pred = forest.predict(X_test)
y_test_scores = forest.decision_function(X_test)

with plt.style.context("dark_background"):
    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_test_pred,
        display_labels=["not anomaly", "anomaly"],
        colorbar=False,
        ax=ax,
    )

    plt.grid(False)
    plt.show()

matrix = confusion_matrix(y_test, y_test_pred)
print(matrix)
tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
print(f"TNR = {tnr}")
print(f"TPR = {tpr}")
print(f"BA = {(tnr + tpr) / 2}")

fpr, tpr, thr = roc_curve(y_test, y_test_scores)
auc_val = roc_auc_score(y_test, y_test_scores)
print(f"ROC-AUC = {auc_val}")

with plt.style.context("dark_background"):
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()