import numpy as np
import matplotlib.pyplot as plt

def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100, batch_size=1):
    m, n = X.shape
    theta = np.zeros((n, 1))
    loss_history = []

    for _ in range(epochs):
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]

        predictions = X_batch.dot(theta)
        errors = predictions - y_batch
        gradient = X_batch.T.dot(errors) / batch_size
        theta -= learning_rate * gradient
        loss = np.mean(errors**2) / 2
        loss_history.append(loss)

    return theta, loss_history

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# Add a bias term to the feature matrix
X_b = np.c_[np.ones((1000, 1)), X]

# Run stochastic gradient descent
theta_sgd, loss_history_sgd = stochastic_gradient_descent(X_b, y, epochs=1000, batch_size=10)

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history_sgd, label='Stochastic Gradient Descent')
plt.title('Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()