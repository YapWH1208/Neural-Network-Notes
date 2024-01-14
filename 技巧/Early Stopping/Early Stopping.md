**Early stopping** is a machine learning technique commonly used to prevent overfitting of a model during training. It will monitor the performance of the model on a validation set during training and stop the training process once the performance stops improving or starts degrading over the threshold.

There are some key components of early stopping:
1. **Monitoring metrics**: The key performance metrics used to monitor training, normally validation loss or accuracy is used.
2. **Patience**: The number of epochs to eait for an improvement over the best performance. This is the threshold of the early stopping technique to stop the training process when the model is not improving.
3. **Saving best model**: When the best performance is tested out, the model will saved the best state so when the early stopping is triggered we still have the model with the best performance.


# Code
## Import Necessary Library
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
```

## Setup CUDA Device
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Build Neural Network
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
```

## Get Data
```python
# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to GPU if available
X_train, y_train = torch.Tensor(X_train).to(device), torch.Tensor(y_train).view(-1, 1).to(device)
X_test, y_test = torch.Tensor(X_test).to(device), torch.Tensor(y_test).view(-1, 1).to(device)
```

![[Data.png|center|650]]
<center>Diagram 1: Data Distribution</center>

## Build Model
```python
# Define the neural network, loss function, and optimizer and move to GPU
input_size = X_train.shape[1]
hidden_size1, hidden_size2, output_size = 32, 16, 1

model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Build Early Stopping Mechanism
```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.epoch_count = 0

    def __call__(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        self.epoch_count += 1
```


## Train
```python
epochs = 1000
early_stopping = EarlyStopping(patience=5)

train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation loss
    model.eval()
    val_outputs = model(X_test)
    val_loss = criterion(val_outputs, y_test)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print("Epoch: {:04d}/{} | Train loss: {:.4f} | Validation loss: {:.4f}".format(epoch + 1, epochs, loss.item(), val_loss.item()))

    # Check early stopping
    early_stopping(val_loss.item())
    if early_stopping.early_stop:
        print("Early stopping!")
        break
```

## Result
```bash
Epoch: 0010/1000 | Train loss: 0.6796 | Validation loss: 0.6737
Epoch: 0020/1000 | Train loss: 0.6558 | Validation loss: 0.6555
Epoch: 0030/1000 | Train loss: 0.6233 | Validation loss: 0.6319
Epoch: 0040/1000 | Train loss: 0.5818 | Validation loss: 0.6017
Epoch: 0050/1000 | Train loss: 0.5328 | Validation loss: 0.5663
Epoch: 0060/1000 | Train loss: 0.4814 | Validation loss: 0.5290
Epoch: 0070/1000 | Train loss: 0.4324 | Validation loss: 0.4927
Epoch: 0080/1000 | Train loss: 0.3889 | Validation loss: 0.4599
Epoch: 0090/1000 | Train loss: 0.3517 | Validation loss: 0.4318
Epoch: 0100/1000 | Train loss: 0.3208 | Validation loss: 0.4093
Epoch: 0110/1000 | Train loss: 0.2962 | Validation loss: 0.3928
Epoch: 0120/1000 | Train loss: 0.2771 | Validation loss: 0.3820
Epoch: 0130/1000 | Train loss: 0.2617 | Validation loss: 0.3746
Epoch: 0140/1000 | Train loss: 0.2491 | Validation loss: 0.3700
Epoch: 0150/1000 | Train loss: 0.2385 | Validation loss: 0.3666
Epoch: 0160/1000 | Train loss: 0.2291 | Validation loss: 0.3632
Epoch: 0170/1000 | Train loss: 0.2204 | Validation loss: 0.3605
Epoch: 0180/1000 | Train loss: 0.2122 | Validation loss: 0.3580
Epoch: 0190/1000 | Train loss: 0.2041 | Validation loss: 0.3560
Epoch: 0200/1000 | Train loss: 0.1962 | Validation loss: 0.3538
Epoch: 0210/1000 | Train loss: 0.1888 | Validation loss: 0.3522
Epoch: 0220/1000 | Train loss: 0.1817 | Validation loss: 0.3504
Epoch: 0230/1000 | Train loss: 0.1748 | Validation loss: 0.3491
Epoch: 0240/1000 | Train loss: 0.1682 | Validation loss: 0.3488
Early stopping!
```

![[Result.png|center|650]]
<center>Diagram 2: Graph of Train Loss against Validation Loss</center>