from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.data import generate_data_clusters
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

X, y = make_blobs(n_samples=[200, 100], n_features=2, centers=[(-10, -10), (10, 10)],
    cluster_std=[2.0, 6.0], random_state=42)

for k in range(5, 55, 5):
    model = KNN(0.1, n_neighbors=k, metric="euclidean")
    model.fit(X)
    train_preds = model.labels_

    model_lof = LOF(contamination=0.1, n_neighbors=k, metric="euclidean")
    model_lof.fit(X)
    lof_preds = model_lof.labels_

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], c=train_preds, cmap='coolwarm', alpha=0.7)
    axes[0].set_title(f"KNN (k = {k})")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    axes[1].scatter(X[:, 0], X[:, 1], c=lof_preds, cmap='coolwarm', alpha=0.7)
    axes[1].set_title(f"LOF (k = {k})")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    fig.suptitle(f"Comparison at k = {k}", fontsize=14)
    plt.tight_layout()
    plt.show()
