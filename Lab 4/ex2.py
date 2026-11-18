import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization
from sklearn.svm import OneClassSVM

mat = loadmat('cardio.mat')
X = mat['X']
y = 1 - 2 * (mat['y'].ravel())
contamination = np.mean(y == 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, random_state=42, stratify=y)
contamination_train = np.mean(y_train == -1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])

param_grid = {
    'ocsvm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'ocsvm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
    'ocsvm__nu': [0.05, 0.1, contamination_train, 0.2]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='balanced_accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)

print("Best Parameters:", grid_search.best_params_)
print("Test Balanced Accuracy:", test_balanced_accuracy)