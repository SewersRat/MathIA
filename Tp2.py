import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])

w = np.zeros(2)
b = 0
lr = 0.1
epochs = 20

def step(x):
    return 1 if x >= 0.5 else 0

print("=== Perceptron Simple ===\n")
for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        y_hat = step(np.dot(X[i], w) + b)
        error = y[i] - y_hat
        w += lr * error * X[i]
        b += lr * error
        errors += int(error != 0)
    print(f"Epoch {epoch+1:2d} | Erreurs: {errors} | w={w} | b={b:.2f}")

print("\n=== Résultats Perceptron ===")
for i in range(len(X)):
    y_hat = step(np.dot(X[i], w) + b)
    print(f"x={X[i]} → prédit: {y_hat} | attendu: {y[i]} | {'✓' if y_hat == y[i] else '✗'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for i in range(len(X)):
    color = 'blue' if y[i] == 0 else 'red'
    ax.scatter(X[i][0], X[i][1], c=color, s=200, zorder=5)
    ax.annotate(f'({X[i][0]},{X[i][1]}) y={y[i]}', X[i], textcoords="offset points", xytext=(10,5))

x_vals = np.linspace(-0.5, 1.5, 100)
ax.plot(x_vals, 0.5 - x_vals, 'g--', label='tentative droite')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('XOR — non linéairement séparable')
ax.legend()
ax.grid(True)

y_mlp = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

np.random.seed(42)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

lr_mlp = 0.1
epochs_mlp = 10000
losses = []

for epoch in range(epochs_mlp):
    h = sigmoid(X @ W1 + b1)
    y_hat = sigmoid(h @ W2 + b2)

    loss = np.mean((y_mlp - y_hat) ** 2)
    losses.append(loss)

    d2 = (y_hat - y_mlp) * sigmoid_deriv(y_hat)
    d1 = (d2 @ W2.T) * sigmoid_deriv(h)

    W2 -= lr_mlp * h.T @ d2
    b2 -= lr_mlp * d2.sum(axis=0)
    W1 -= lr_mlp * X.T @ d1
    b1 -= lr_mlp * d1.sum(axis=0)

print("\n=== Résultats MLP (couche cachée) ===")
for i in range(len(X)):
    print(f"x={X[i]} → {y_hat[i][0]:.4f} → prédit: {int(y_hat[i][0] > 0.5)} | attendu: {y_mlp[i][0]}")

ax2 = axes[1]
ax2.plot(losses)
ax2.set_title('MLP — Courbe de perte')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.grid(True)

plt.tight_layout()
plt.show()
print("\nGraphiques sauvegardés.")