# torch.argmax
- Used to find the indices of the maximum value in the tensor

```python
import torch

x = torch.tensor([[1,2,3],
				  [4,5,6]])

print(torch.argmax(x, dim=1))
```

- `dim=1`: It will serach along second dimension which will be the column dimension in this case

Output:

```
tensor([2,2])
```

# torch.max
- Used to find the maximum values in the tensor
- It can return both the maximum value and also the indices of the maximum value in the tensor

```python
import torch

x = torch.tensor([[1,2,3],
				  [4,5,6]])

values, indices = torch.max(x, dim=1)
print(values)
print(indices)
```

Output:

```
tensor([3,6])
tensor([2,2])
```

