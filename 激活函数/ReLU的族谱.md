#relu 
#gelu
#leaky_relu
#softplus

# ReLU 分叉树
## ReLU
$$\text{ReLU}(x) = \begin{cases} 0 & \text{if } x < 0 \\ x & \text{if } x \geq 0 \end{cases}$$
- 从这个表达式中，我们可以看到一个问题，即ReLU激活函数会将负值更改为0。这可能会导致更新后的值保持为0并且在反向传播更新时没有任何改进。

## GELU
$$\text{GELU}(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right )\right)$$
- GELU 作为 ReLU 的一个分支。 它通过使用高斯分布来改变 0 值。
- 但在这种特殊情况下，这些值被标准化为$[0,1]$的范围，这将导致使用GELU后的值仍然接近0。

## Leaky ReLU
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}$$
- Leaky ReLU 通过改为 $\alpha x$ 解决了 ReLU 的“死”神经元问题
- 但这会导致用户需要将多一个超参数作为变量进行调整。

## Softplus
$$\text{Softplus}(x) = \log(1 + e^x)$$
- 此外，作为 ReLU 激活函数的一个分支，它通过让它变得更平滑来解决 ReLU 的问题
- 但它添加了类似于sigmoid激活函数的指数项，这会导致计算过程变慢。

# 参考文献
- [GELU激活函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/394465965)
- [速用笔记 | Sigmoid/Tanh/ReLu/Softplus 激活函数的图形、表达式、导数、适用条件 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/461707201)
