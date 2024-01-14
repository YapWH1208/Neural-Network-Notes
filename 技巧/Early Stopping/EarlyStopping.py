import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network architecture
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

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to GPU if available
X_train, y_train = torch.Tensor(X_train).to(device), torch.Tensor(y_train).view(-1, 1).to(device)
X_test, y_test = torch.Tensor(X_test).to(device), torch.Tensor(y_test).view(-1, 1).to(device)

# Define the neural network, loss function, and optimizer and move to GPU
input_size = X_train.shape[1]
hidden_size1, hidden_size2, output_size = 32, 16, 1

model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define early stopping
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

# Train the model with early stopping
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

# Plot loss diagram
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.scatter(early_stopping.epoch_count, val_losses[early_stopping.epoch_count - 1], color='red', label='Early Stopping Point')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot data distribution
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Data Distribution')
plt.show()