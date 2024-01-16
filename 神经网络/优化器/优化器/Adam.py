import numpy as np
import matplotlib.pyplot as plt

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

# Run Adam optimizer
theta_adam, loss_history_adam = adam_optimizer(X_b, y, epochs=1000)

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history_adam, label='Adam Optimizer')
plt.title('Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()