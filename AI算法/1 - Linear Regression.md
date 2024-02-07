# Theory
## Regression
- Representing the relationship between instances using a line.

## Linear
- Instances appear to be aligned in a straight line.
- This line may not necessarily be two-dimensional; it can be a line in higher dimensions.

## Linear Regression
- Expressing the relationship between instances using a straight line.
- Mainly categorized into: Simple Linear Regression, Multiple Linear Regression.

### Simple Linear Regression
- Evolved from a linear equation $$\large y = wx +b$$
- Utilizes machine learning concept parameters $w$ and $b$ $$\large \hat y = w_0 + w_1x + \epsilon$$
### Multiple Linear Regression
- Involves more features $$\large \hat y_i = w_0 + w_1x_1 + w_2x_2 + ...+w_ix_i$$
- By utilizing the matrix form expression $$\large \hat y = X^Tw$$
## Loss Function
- How do we determine the most suitable line when seeking it?
	- The simplest method is using the formula of squared differences.
	- More accurately, it is the residual sum of squares.

## Least Squares Method
- Derived from [Residual Sum of Squares (RSS), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Standard Deviation (SD)].
- For simple linear regression, it still maintains the basic formula of residual sum of squares.
- Whereas for multiple linear regression, expressed in matrix form, it is $$\large SSE = (y -Xw)^T(y - Xw)$$ This form seeks the residual sum of squares for a line.
- To find the most suitable line, taking derivatives to find the extreme value, we get $$\large w = (X^TX)^{-1}X^Ty$$ However, when our instance matrix is not a full-rank matrix, we encounter situations where the least squares solution is not a unique solution, hence optimizing this algorithm.

# Variations
## Ridge Regression
- To make the instance matrix invertible, we add a unit matrix $I$ within it, yielding the following new formula $$\large w = (X^TX+\lambda I)^{-1}X^Ty$$
- A necessary condition for an invertible matrix is the absence of multicollinearity (multiple solutions simultaneously).
- This method is also known as L2 regularization.

## Lasso Regression
- Unlike L2 regularization, Lasso regression adds a constrained loss function, resulting in $$\large w = (X^TX)^{-1}X^Ty + \lambda\sum|w_i|$$
- This method is referred to as L1 regularization.

# References
- [Machine Learning | Algorithm Notes - Linear Regression](https://zhuanlan.zhihu.com/p/139445419)
- [Explaining Linear Regression in Plain Language](https://zhuanlan.zhihu.com/p/72513104)