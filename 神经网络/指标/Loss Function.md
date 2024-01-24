The loss function is used to evaluate the extent to which a model's predictions differ from the actual values. A lower loss function generally indicates better model performance. Different models often use different loss functions.

Loss functions can be categorized into empirical risk loss functions and structural risk loss functions. Empirical risk loss functions measure the difference between predicted and actual results, while structural risk loss functions include a regularization term added to the empirical risk loss function.

# 0-1 (Zero-one) Loss Function
If predicted value and the target value is not equivalent then $1$ else $0$ .$$L(Y,f(X))=\begin{cases}1&Y\neq f(X)\\0&Y=f(X)\end{cases}$$
- The 0-1 loss function directly corresponds to the number of classification judgment errors. However, it is a non-convex function and is not very suitable.
- The perceptron uses this type of loss function. However, the condition of exact equality is too strict, so the condition can be relaxed. That is, it is considered equal when $|Y-f(X)|<T$ $$L(Y,f(X))=\begin{cases}1&|Y-f(X)|\geqslant T\\0&|Y-f(X)|<T\end{cases}$$
# Absolute Loss Function
The absolute loss function calculates the absolute difference between the predicted value and the target value:$$L(Y,f(X))=|Y-f(X)|$$
# Logarithmic Loss Function
The standard form of the logarithmic loss function is as follows: $$L(Y,P(Y|X))=-\log P(Y|X)$$
- The logarithmic loss function is particularly effective in representing probability distributions. In many scenarios, especially in multi-class classification, if there is a need to know the confidence of the result belonging to each class, it is well-suited.
- It is not very robust and is more sensitive to noise compared to the hinge loss.
- The loss function for logistic regression is the logarithmic loss function.

# Mean Squared Error (MSE) Loss Function
The standard form of the mean squared error (MSE) loss function is as follows:$$L(Y|F(X))=\sum_N(Y-f(X))^2$$
- Commonly used in regression problem

# Exponential Loss Function
The standard form of the exponential loss function is as follows:$$L(Y|f(X))=\exp[-yf(x)]$$
- Highly sensitive to outliers and noise. It is often used in AdaBoost algorithms.

# Hinge Loss Function
The standard form of the hinge loss function is as follows: $$L(Y,f(X))=\max(0,1-yf(x))$$
- The hinge loss function indicates that if a sample is classified correctly, the loss is $0$; otherwise, the loss is given by $1-yf(x)$. Support Vector Machines (SVM) use this loss function.
- Typically, $f(X)$​ represents the predicted value, which should fall between -1 and 1, and $Y$ represents the target value (-1 or 1). This implies that the value of $f(X)$​ between -1 and +1 is sufficient. It does not encourage extreme values of $f(X)$. In other words, the classifier is not encouraged to be overly confident. There is no reward for a correctly classified sample being further than 1 unit away from the decision boundary. This encourages the classifier to focus on the overall error.
- Hinge loss is relatively robust, less sensitive to outliers and noise. However, it lacks a clear probability interpretation.

# Perceptron Loss Function
The standard form of the perceptron loss function is as follows: $$L(Y,f(X))=\max(0,-f(x))$$
- Perceptron Loss is a variant of the Hinge Loss function. While Hinge Loss strongly penalizes points near the decision boundary (correct side), Perceptron Loss is satisfied as long as the sample is correctly classified, regardless of its distance from the decision boundary. It is simpler than Hinge Loss because it does not involve a max-margin boundary. Consequently, models using Perceptron Loss may not have the same level of generalization ability as those using Hinge Loss, as it lacks the max-margin characteristic that enhances generalization in Hinge Loss.

# Cross-Entropy Loss Function
The standard form of the cross-entropy loss function is as follows: $$C=-\frac1N\sum_x[y\ln a+(1-y)\ln(1-a)]$$
which $x$ is the sample, $y$ is the target label, $a$ is the predicted output, $n$ is the number of samples.

- It is essentially a type of log-likelihood function that can be used for both binary and multi-class tasks.

	For binary classification problems, when used as a loss function (assuming the input data is the output of softmax or sigmoid functions), the formula is:$$\text{loss}=-\frac1N\sum_x[y\ln a+(1-y)\ln(1-a)]$$For multi-class classification problems, the formula is:$$\text{loss}=-\frac1N\sum_iy_i\ln a_i$$
- When using the sigmoid activation function, the cross-entropy loss function is commonly used instead of the mean squared error loss function. This choice is made because cross-entropy effectively addresses the issue of slow weight updates associated with the mean squared error loss function. Cross-entropy exhibits a favorable property of "fast weight updates when the error is large and slow updates when the error is small." This property makes it well-suited for training models with the sigmoid activation function.

# Reference
- [常见的损失函数(loss function)总结](https://zhuanlan.zhihu.com/p/58883095)