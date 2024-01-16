import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros((n, 1))
    loss_history = []

    for _ in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / m
        theta -= learning_rate * gradient
        loss = np.mean(errors**2) / 2
        loss_history.append(loss)

    return theta, loss_history

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to the feature matrix
X_b = np.c_[np.ones((100, 1)), X]

# Run gradient descent
theta_gd, loss_history_gd = gradient_descent(X_b, y, epochs=1000)

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history_gd, label='Gradient Descent')
plt.title('Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()