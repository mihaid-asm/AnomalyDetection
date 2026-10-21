import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

np.random.seed(42)
dim = 3
n_samples = 1000
contamination = 0.1
n_outliers = int(n_samples * contamination)
n_inliers = n_samples - n_outliers

mu = np.array([2.0, -1.0, 3.0])

A = np.array([
    [1.5, 0.5, 0.2],
    [0.1, 1.2, -0.4],
    [0.3, 0.2, 0.9]
])
sigma = A @ A.T

L = np.zeros((dim, dim))
for i in range(dim):
    for j in range(i + 1):
        s = np.sum(L[i, :j] * L[j, :j])
        if i == j:
            L[i, j] = np.sqrt(sigma[i, i] - s)
        else:
            L[i, j] = (sigma[i, j] - s) / L[j, j]

x_std_inliers = np.random.normal(0, 1, size=(n_inliers, dim))
y_inliers = (L @ x_std_inliers.T).T + mu

y_outliers = np.random.uniform(low=-15, high=15, size=(n_outliers, dim))

X = np.vstack([y_inliers, y_outliers])
y_true = np.concatenate([np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)])

diff = X - mu
z = np.linalg.solve(L, diff.T).T
z_scores = np.linalg.norm(z, axis=1)

z_threshold = np.quantile(z_scores, 1.0 - contamination)
y_pred = (z_scores > z_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])
tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
ba = (tnr + tpr) / 2

print(f"TNR: {tnr:.4f}")
print(f"TPR: {tpr:.4f}")
print(f"Balanced Accuracy: {ba:.4f}")
print(f"sklearn BA check: {balanced_accuracy_score(y_true, y_pred):.4f}")