Some common used learning rate schedular in Pytorch:

# StepLR

```python
torch.optim.lr_scheduler.StepLR(optimizer,step_size,gamma=0.1,last_epoch=-1,verbose=False)
```

Changes the learning rate according the `step_size` with multiplier `gamma`

Parameter:
- `step_size` (int) : Number of epochs after changing the learning rate
- `gamma` (float) : learning rate multiplier
- `last_epoch` (int) : The index of last epoch to change the learning rate

![[StepLR.png|center|650]]
<center>step_size=5, gamma=0.5, init_lr=0.1</center>

# MultiStepLR

```python
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)
```

Changes the learning rate according the given index of epochs

Parameter:
- `milestones` (list) : A list containing the index of epochs to change the learning rate. The list must be in increment. For example, `[20,50,100]` will change the learning rate at epoch 20,50,100.
- `gamma` (float) : learning rate multiplier
- `last_epoch` (int) : The index of last epoch to change the learning rate

![[MultiStepLR.png|center|650]]
<center>milestones=[5,10,15], gamma=0.5, init_lr=0.1</center>

# ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
```

Adjust learning rate by exponential decay. Adjusting formula: $\text{lr}*\text{gamma}^\text{{epoch}}$

Parameter:
- `gamma` (float) : learning rate multiplier
- `last_epoch` (int) : The index of last epoch to change the learning rate

![[ExponentialLR.png|center|650]]
<center>gamma=0.5, init_lr=0.1</center>

# CosineAnnealingLR

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)
```

Use cosine annealing to adjust learning rate

Parameter:
- `T_max` (int) : Maximum number of iterations
- `eta_max` (int) : Minimum learning rate
- `last_epoch` (int) : The index of last epoch to change the learning rate

![[CosineAnnealingLR.png|center|650]]
<center>T_max=10, eta_min=0, init_lr=0.1</center>