

import numpy as np

# --- Données XOR ---
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# --- Fonctions d'activation ---
sigmoid = lambda x: 1/(1+np.exp(-x))
sigmoid_derivative = lambda x: x*(1-x)

# --- Initialisation ---
np.random.seed(42)
W1, b1 = np.random.randn(2,2), np.zeros((1,2))
W2, b2 = np.random.randn(2,1), np.zeros((1,1))
lr = 0.1

# --- Entraînement compact ---
for _ in range(10000):
    h = sigmoid(X@W1 + b1)
    out = sigmoid(h@W2 + b2)
    d_out = (y - out) * sigmoid_derivative(out)
    d_h = d_out@W2.T * sigmoid_derivative(h)
    W2 += h.T @ d_out * lr; b2 += d_out.sum(0,keepdims=True) * lr
    W1 += X.T @ d_h * lr; b1 += d_h.sum(0,keepdims=True) * lr

# --- Résultat ---
print(np.round(out))