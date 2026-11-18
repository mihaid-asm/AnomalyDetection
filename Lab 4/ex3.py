import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import OneClassSVM
from pyod.models.deep_svdd import DeepSVDD

df_train = pd.read_csv("shuttle.trn", header=None, delimiter=r"\s+")
df_test = pd.read_csv("shuttle.tst", header=None, delimiter=r"\s+")

X_train = df_train.iloc[:, :-1].values
y_train_raw = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test_raw = df_test.iloc[:, -1].values

y_train = np.where(y_train_raw == 1, 0, 1)
y_test = np.where(y_test_raw == 1, 0, 1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
contamination = np.mean(y_train == 1)

print("OCSVM")
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination)
ocsvm.fit(X_train_scaled)

y_test_pred_ocsvm = np.where(ocsvm.predict(X_test_scaled) == 1, 0, 1)
y_test_scores_ocsvm = -ocsvm.decision_function(X_test_scaled)

ba_ocsvm = balanced_accuracy_score(y_test, y_test_pred_ocsvm)
roc_ocsvm = roc_auc_score(y_test, y_test_scores_ocsvm)
print(f"OCSVM:\n Balanced Accuracy: {ba_ocsvm:.4f}\n ROC AUC: {roc_ocsvm:.4f}\n")

architectures = [
    [32, 16],
    [64, 32, 16]
]

for arch in architectures:
    print(f"Training DeepSVDD with hidden_neurons={arch}")
    deep_svdd = DeepSVDD(
        n_features=X_train_scaled.shape[1],
        hidden_neurons=arch,
        epochs=15,
        contamination=contamination,
        random_state=42
    )

    deep_svdd.fit(X_train_scaled)

    y_test_pred_ds = deep_svdd.predict(X_test_scaled)
    y_test_scores_ds = deep_svdd.decision_function(X_test_scaled)

    ba_ds = balanced_accuracy_score(y_test, y_test_pred_ds)
    roc_ds = roc_auc_score(y_test, y_test_scores_ds)

    print(f"DeepSVDD {arch}:\n Balanced Accuracy: {ba_ds:.4f}\n ROC AUC: {roc_ds:.4f}\n")