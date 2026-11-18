from pyod.utils.data import generate_data_clusters
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=300, n_test=200, n_features=3, n_clusters=2, contamination=0.15, random_state=67)

clf = OCSVM(kernel='rbf', contamination=0.15)
clf.fit(X_train)

# LINEAR
# Balanced Accuracy: 0.41369047619047616
# ROC AUC: 0.050967261904761904

# RBF
# Balanced Accuracy: 0.9598214285714286
# ROC AUC: 0.9985119047619048

y_train_pred = clf.labels_
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

bal_acc = balanced_accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_scores)

print(f"Balanced Accuracy: {bal_acc}")
print(f"ROC AUC: {roc_auc}")

fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='red', label='Outlier', alpha=0.7)
ax1.set_title('Train Ground Truth')
ax1.legend()

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_train[y_train_pred == 0, 0], X_train[y_train_pred == 0, 1], X_train[y_train_pred == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax2.scatter(X_train[y_train_pred == 1, 0], X_train[y_train_pred == 1, 1], X_train[y_train_pred == 1, 2], c='red', label='Outlier', alpha=0.7)
ax2.set_title('Train Predicted')
ax2.legend()

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax3.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], c='red', label='Outlier', alpha=0.7)
ax3.set_title('Test Ground Truth')
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[y_test_pred == 0, 0], X_test[y_test_pred == 0, 1], X_test[y_test_pred == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax4.scatter(X_test[y_test_pred == 1, 0], X_test[y_test_pred == 1, 1], X_test[y_test_pred == 1, 2], c='red', label='Outlier', alpha=0.7)
ax4.set_title('Test Predicted')
ax4.legend()

plt.tight_layout()
plt.show()



clf = DeepSVDD(n_features=3, contamination=0.15)
clf.fit(X_train)

y_train_pred = clf.labels_
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

bal_acc = balanced_accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_scores)

print(f"DeepSVDD Balanced Accuracy: {bal_acc}")
print(f"DeepSVDD ROC AUC: {roc_auc}")

fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='red', label='Outlier', alpha=0.7)
ax1.set_title('Train Ground Truth')
ax1.legend()

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_train[y_train_pred == 0, 0], X_train[y_train_pred == 0, 1], X_train[y_train_pred == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax2.scatter(X_train[y_train_pred == 1, 0], X_train[y_train_pred == 1, 1], X_train[y_train_pred == 1, 2], c='red', label='Outlier', alpha=0.7)
ax2.set_title('Train Predicted')
ax2.legend()

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax3.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], c='red', label='Outlier', alpha=0.7)
ax3.set_title('Test Ground Truth')
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[y_test_pred == 0, 0], X_test[y_test_pred == 0, 1], X_test[y_test_pred == 0, 2], c='blue', label='Inlier', alpha=0.7)
ax4.scatter(X_test[y_test_pred == 1, 0], X_test[y_test_pred == 1, 1], X_test[y_test_pred == 1, 2], c='red', label='Outlier', alpha=0.7)
ax4.set_title('Test Predicted')
ax4.legend()

plt.tight_layout()
plt.show()
