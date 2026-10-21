import numpy as np
import matplotlib.pyplot as plt

def compute_leverage(X):
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    H = X_design @ np.linalg.pinv(X_design.T @ X_design) @ X_design.T
    return np.diag(H)

def generate_points_1d(a, b, sigma_y, n_regular=150, n_out=10):
    x_reg = np.random.normal(loc=0.0, scale=1.0, size=n_regular)
    y_reg = a * x_reg + b + np.random.normal(loc=0.0, scale=sigma_y, size=n_regular)
    x_high_x = np.random.normal(loc=0.0, scale=5.0, size=n_out)
    y_high_x = a * x_high_x + b + np.random.normal(loc=0.0, scale=sigma_y, size=n_out)
    x_high_y = np.random.normal(loc=0.0, scale=1.0, size=n_out)
    y_high_y = a * x_high_y + b + np.random.normal(loc=0.0, scale=5.0 * sigma_y + 10.0, size=n_out)
    x_high_xy = np.random.normal(loc=0.0, scale=5.0, size=n_out)
    y_high_xy = a * x_high_xy + b + np.random.normal(loc=0.0, scale=5.0 * sigma_y + 10.0, size=n_out)

    x = np.concatenate([x_reg, x_high_x, x_high_y, x_high_xy])
    y = np.concatenate([y_reg, y_high_x, y_high_y, y_high_xy])
    types = (
            ['Regular'] * n_regular +
            ['High X'] * n_out +
            ['High Y'] * n_out +
            ['High X & Y'] * n_out
    )
    return x, y, np.array(types)

a_1d, b_1d = 2.0, 1.0
sigmas_1d = [0.5, 2.0, 5.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, s in enumerate(sigmas_1d):
    x, y, types = generate_points_1d(a_1d, b_1d, s)
    h = compute_leverage(x[:, np.newaxis])
    top_indices = np.argsort(h)[-10:]
    ax = axes[idx]
    for category, color in [('Regular', 'blue'), ('High X', 'orange'), ('High Y', 'green'), ('High X & Y', 'purple')]:
        mask = types == category
        ax.scatter(x[mask], y[mask], label=category, alpha=0.5, s=25, color=color)
    ax.scatter(x[top_indices], y[top_indices], facecolors='none', edgecolors='red', s=120, linewidth=2,label='Top Leverage')
    ax.set_title(f"1D Model: $\\sigma^2={s ** 2:.2f}$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$y$")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

def generate_points_2d(a, b, c, sigma_y, n_regular=200, n_out=15):
    X_reg = np.random.normal(loc=0.0, scale=1.0, size=(n_regular, 2))
    y_reg = a * X_reg[:, 0] + b * X_reg[:, 1] + c + np.random.normal(loc=0.0, scale=sigma_y, size=n_regular)
    X_high_x = np.random.normal(loc=0.0, scale=5.0, size=(n_out, 2))
    y_high_x = a * X_high_x[:, 0] + b * X_high_x[:, 1] + c + np.random.normal(loc=0.0, scale=sigma_y, size=n_out)
    X_high_y = np.random.normal(loc=0.0, scale=1.0, size=(n_out, 2))
    y_high_y = (a * X_high_y[:, 0] + b * X_high_y[:, 1] + c +
                np.random.normal(loc=0.0, scale=5.0 * sigma_y + 10.0, size=n_out))
    X_high_xy = np.random.normal(loc=0.0, scale=5.0, size=(n_out, 2))
    y_high_xy = (a * X_high_xy[:, 0] + b * X_high_xy[:, 1] + c +
                 np.random.normal(loc=0.0, scale=5.0 * sigma_y + 10.0,size=n_out))

    X = np.vstack([X_reg, X_high_x, X_high_y, X_high_xy])
    y = np.concatenate([y_reg, y_high_x, y_high_y, y_high_xy])
    types = (
            ['Regular'] * n_regular +
            ['High X'] * n_out +
            ['High Y'] * n_out +
            ['High X & Y'] * n_out
    )
    return X, y, np.array(types)

a_2d, b_2d, c_2d = 1.5, -2.0, 0.5
sigmas_2d = [0.5, 2.0, 5.0]
fig = plt.figure(figsize=(18, 5))
for idx, s in enumerate(sigmas_2d):
    X, y, types = generate_points_2d(a_2d, b_2d, c_2d, s)
    h = compute_leverage(X)
    top_indices = np.argsort(h)[-15:]
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
    for category, color in [('Regular', 'blue'), ('High X', 'orange'), ('High Y', 'green'), ('High X & Y', 'purple')]:
        mask = types == category
        ax.scatter(X[mask, 0], X[mask, 1], y[mask], label=category, alpha=0.5, s=20, color=color)
    ax.scatter(X[top_indices, 0], X[top_indices, 1], y[top_indices], facecolors='none', edgecolors='red', s=80, linewidth=2, label='Top Leverage')
    ax.set_title(f"3D Space: $\\sigma^2={s**2:.2f}$")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    ax.legend()

plt.tight_layout()
plt.show()
