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

def adam_optimizer(X, y, learning_rate=0.01, epochs=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, n = X.shape
    theta = np.zeros((n, 1))
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    t = 0
    loss_history = []

    for epoch in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors)
        
        t += 1
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)

        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)

        theta -= learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

        loss = np.mean(errors**2) / 2
        loss_history.append(loss)

    return theta, loss_history

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# Add a bias term to the feature matrix
X_b = np.c_[np.ones((1000, 1)), X]

# Run gradient descent
theta_gd, loss_history_gd = gradient_descent(X_b, y, epochs=1000)

# Run stochastic gradient descent
theta_sgd, loss_history_sgd = stochastic_gradient_descent(X_b, y, epochs=1000, batch_size=10)

# Run Adam optimizer
theta_adam, loss_history_adam = adam_optimizer(X_b, y, epochs=1000)

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history_gd, label='Gradient Descent')
plt.plot(loss_history_sgd, label='Stochastic Gradient Descent')
plt.plot(loss_history_adam, label='Adam Optimizer')
plt.title('Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()