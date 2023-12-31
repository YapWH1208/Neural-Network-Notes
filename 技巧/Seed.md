```python
import torch
import os
import numpy as np
import random

def setup_seed(seed:int=42):
	os.environ['PYTHONHASHSEED'] = str(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	np.random.seed(seed)
	random.seed(seed)

	torch.backends.cudnn.deterministics = True
	torch.backends.cudnn.benchmarks = False
	torch.backends.cudnn.enabled = False
```
<center>Code for seed setup</center>

# Advantages:
- Setting up seed can help in getting the same results in experiments which is good for ablation experiment

# Disadvantages:
- Locking seed will decrease the randomness of the output
- Not suitable for application

# Reference
- [The story behind ‘random.seed(42)’ in machine learning](https://medium.com/geekculture/the-story-behind-random-seed-42-in-machine-learning-b838c4ac290a)
- [深度学习中random.seed(42)背后的故事](https://zhuanlan.zhihu.com/p/458809368)
