import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_data():
    data = loadmat("mnistdata.mat")
    X = data["X"]
    y = data["y"].flatten()
    return X, y

def load_weights():
    weights = loadmat("pesos.mat")
    W2 = weights["W2"]
    b2 = weights["b2"]
    W3 = weights["W3"]
    b3 = weights["b3"]

    return W2, b2, W3, b3

print("Carregando dados...")
X, y = load_data()

imagem = X[0,:].reshape(20,20).T
plt.imshow(imagem, cmap='gray')
plt.colorbar(label="Intensidade do Pixel")
plt.show()

def display_random_images(X, y):
    indices = np.random.choice(X.shape[0], 100, replace=False)
    images = X[indices, :].reshape(-1, 20, 20).transpose(0, 2, 1)
    labels = y[indices]

    fig, axes = plt.subplots(10, 10, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"{labels[i]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    display_random_images(X, y)

print("Carregando pesos treinados...")
W2, b2, W3, b3 = load_weights()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(W2, b2, W3, b3, X):
    z2 = X @ W2.T + b2
    a2 = sigmoid(z2)

    z3 = a2 @ W3.T + b3
    a3 = sigmoid(z3)
    return a3.argmax(axis=1) + 1