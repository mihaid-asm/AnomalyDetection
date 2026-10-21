from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

X_train, X_test, y_train, y_test = generate_data_clusters(400, 200, 2, 2, 0.1)

for k in range(5, 55, 5):
    model = KNN(0.1, n_neighbors=k, metric="euclidean")
    model.fit(X_train)
    train_preds = model.labels_
    test_preds = model.predict(X_test)

    train_ba = balanced_accuracy_score(y_train, train_preds)
    test_ba = balanced_accuracy_score(y_test, test_preds)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
    axes[0, 0].set_title("Ground Truth (Train)")

    axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=train_preds, cmap='coolwarm', alpha=0.7)
    axes[0, 1].set_title(f"Predictions (Train) - BA: {train_ba:.3f}")

    axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', alpha=0.7)
    axes[1, 0].set_title("Ground Truth (Test)")

    axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=test_preds, cmap='coolwarm', alpha=0.7)
    axes[1, 1].set_title(f"Predictions (Test) - BA: {test_ba:.3f}")

    fig.suptitle(f"KNN Anomaly Detection (k = {k})", fontsize=14)
    plt.tight_layout()
    plt.show()
