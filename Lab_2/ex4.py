import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

mat = loadmat('cardio.mat')
X = mat['X']
y = mat['y'].ravel()
contamination = np.mean(y == 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_list = np.linspace(30, 120, 10, dtype=int)
train_scores_list = []
test_scores_list = []

for k in k_list:
    clf = LOF(n_neighbors=k, contamination=contamination)
    clf.fit(X_train)

    train_preds = clf.labels_
    test_preds = clf.predict(X_test)
    train_ba = balanced_accuracy_score(y_train, train_preds)
    test_ba = balanced_accuracy_score(y_test, test_preds)

    print(f"k = {k:3d} | Train BA: {train_ba:.4f} | Test BA: {test_ba:.4f}")

    train_scores_list.append(clf.decision_scores_)
    test_scores_list.append(clf.decision_function(X_test))

train_scores = np.column_stack(train_scores_list)
test_scores = np.column_stack(test_scores_list)

train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

comb_methods = {
    "Average": average,
    "Maximization": maximization
}

for name, func in comb_methods.items():
    comb_train_scores = func(train_scores_norm)
    comb_test_scores = func(test_scores_norm)

    threshold = np.quantile(comb_train_scores, 1.0 - contamination)

    comb_train_preds = (comb_train_scores > threshold).astype(int)
    comb_test_preds = (comb_test_scores > threshold).astype(int)

    comb_train_ba = balanced_accuracy_score(y_train, comb_train_preds)
    comb_test_ba = balanced_accuracy_score(y_test, comb_test_preds)

    print(f"\n[{name} Combination]")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train BA : {comb_train_ba:.4f}")
    print(f"Test BA  : {comb_test_ba:.4f}")