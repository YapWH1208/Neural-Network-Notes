import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Define a simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create an instance of the model
model = SimpleModel()

# Define other training parameters
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # StepLR scheduler

# Training loop
num_epochs = 20
learning_rates = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(torch.randn(10))
    loss = criterion(outputs, torch.randn(1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # StepLR scheduler step
    scheduler.step()

    # Append current learning rate to the list for plotting
    learning_rates.append(optimizer.param_groups[0]['lr'])

# Plotting the learning rate changes
plt.plot(range(1, num_epochs+1), learning_rates)
plt.title('Learning Rate Changes with StepLR Scheduler')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()