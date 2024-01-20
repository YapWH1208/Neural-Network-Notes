>盲人下山法👨‍🦯👨‍🦯👨‍🦯——拐杖周围挨着敲一圈，找个最陡的坡下山  
>随机盲人下山法👩‍🦯——爷每次就朝着拐杖第一下随便敲的坡下山  
>小批量盲人下山法👨‍🦯👩‍🦯——爷拐杖周围随便敲几下，从里面找个最陡的坡下

# 梯度下降
## 梯度下降
- 同时使用全部实例点更新参数
- $n$个迭代

## 随机梯度下降
- 使用随机一个实例点更新参数
- 在一个迭代使用所有的实例点
- $m\times n$迭代

## 小批量随机梯度下降
- 使用随机几个实例点更新参数
- ${m\over b}\times n$迭代


# 优缺点

| GD     |  因素        | SGD        |
| ------ | -------- | ---------- |
| 慢     | 速度     | 快         |
| 更准确 | 拟合     | 不那么准确 |
| 小     | 噪声影响 | 大         |
| 大     | 内存需求 | 小         |


# 代码实现
```python
import numpy as np

# Gradient Descent (GD) for matrix data
def gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize parameters
    
    for _ in range(num_iterations):
        # Compute predictions
        predictions = np.dot(X, theta)
        
        # Calculate gradient
        gradient = np.dot(X.T, predictions - y) / m
        
        # Update parameters
        theta -= learning_rate * gradient
    
    return theta

# Stochastic Gradient Descent (SGD) for matrix data
def stochastic_gradient_descent(X, y, learning_rate, num_iterations, batch_size):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize parameters
    
    for _ in range(num_iterations):
        # Randomly shuffle the data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            # Select mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute predictions
            predictions = np.dot(X_batch, theta)
            
            # Calculate gradient
            gradient = np.dot(X_batch.T, predictions - y_batch) / batch_size
            
            # Update parameters
            theta -= learning_rate * gradient
    
    return theta
```