#sigmoid
#softmax
# 理论
- 对于二项分类问题，sigmoid的效果会优于softmax
	- Sigmoid 会将原始数据由$(-\infty,+\infty)$ 映射成 $(0,1)$ 区间的数据，这恰好用于代表概率
- 但对于多项分类问题，常用的方式里就有softmax函数了
	- Softmax函数会将原始数据归一化为 $(0,1)$ 区间的数据，但是它所输出的向量的总和为 $1$ 。
	- 这导致了当面对多项分类预测问题时可以通过最大的概率获得最有可能准确的预测

# 验证
```py
class optim():
    def sigmoid(self, y):
        return 1/(1 + np.exp(-y))
        
    def softmax(self, x):
	    shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
```

- 测试
```py
a = np.array([1,2,3])
b = np.array([[1],[2],[3]])

print(optim().softmax(a))
print(optim().sigmoid(a))
print()
print(optim().softmax(b))
print(optim().sigmoid(b))
```

- 结果：
```py
[0.09003057 0.24472847 0.66524096]
[0.73105858 0.88079708 0.95257413]

[[0.09003057] 
 [0.24472847] 
 [0.66524096]] 
[[0.73105858]
 [0.88079708]
 [0.95257413]]
```

# 参考文献
- [三分钟读懂Softmax函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/168562182#:~:text=Softmax%E6%98%AF%E4%B8%80%E7%A7%8D%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%EF%BC%8C%E5%AE%83%E5%8F%AF%E4%BB%A5%E5%B0%86%E4%B8%80%E4%B8%AA%E6%95%B0%E5%80%BC%E5%90%91%E9%87%8F%E5%BD%92%E4%B8%80%E5%8C%96%E4%B8%BA%E4%B8%80%E4%B8%AA%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83%E5%90%91%E9%87%8F%EF%BC%8C%E4%B8%94%E5%90%84%E4%B8%AA%E6%A6%82%E7%8E%87%E4%B9%8B%E5%92%8C%E4%B8%BA1%E3%80%82,Softmax%E5%8F%AF%E4%BB%A5%E7%94%A8%E6%9D%A5%E4%BD%9C%E4%B8%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E6%9C%80%E5%90%8E%E4%B8%80%E5%B1%82%EF%BC%8C%E7%94%A8%E4%BA%8E%E5%A4%9A%E5%88%86%E7%B1%BB%E9%97%AE%E9%A2%98%E7%9A%84%E8%BE%93%E5%87%BA%E3%80%82%20Softmax%E5%B1%82%E5%B8%B8%E5%B8%B8%E5%92%8C%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E4%B8%80%E8%B5%B7%E7%BB%93%E5%90%88%E4%BD%BF%E7%94%A8%E3%80%82)
- [你 真的 懂 Softmax 吗？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/90771255)

