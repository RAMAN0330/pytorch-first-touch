import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %matplotlib inline

# convert data into tensor of 4D (images, Heights, Weights, Color Channel)
transform = transforms.ToTensor()

# Train Data
train_data = datasets.MNIST(
    root= "/cnn_data",
    download = True,
    train = True,
    transform= transform
)

# Test Data
test_data = datasets.MNIST(
    root= "/cnn_data",
    download = True,
    train = False,
    transform= transform
)

print(train_data, test_data)

# Create a small batch size for images
train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = True)

# Defining Convolutional Model
class CNNModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1,)
    self.conv2 = nn.Conv2d(6,16,3,1)
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self,x):
    X = f.relu(self.conv1(x))
    X = f.max_pool2d(X,2,2)
    X = f.relu(self.conv2(X))
    X = f.max_pool2d(X,2,2)
    X = f.relu(self.fc1(X.view(-1,5*5*16)))
    x = f.relu(self.fc2(X))
    X = self.fc3(x)

    return f.log_softmax(X, dim = 1)

# Create an Instance of our model
torch.manual_seed(41)
model = CNNModel()
print(model.parameters)

print(model.eval())

# Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # Smaller the learning rate, longer it takes to train a model

import time
start_time = time.time()

# Creates to variables to track things
epochs = 5
train_loss = []
test_loss = []
train_correct = []
test_correct = []

for i in range(epochs):
  trn_corr = 0
  tst_corr = 0

  for b, (X_train, y_train) in enumerate(train_loader):
    b+=1 # Start our batches = 1
    y_pred = model(X_train)  # Forward propagation
    loss = criterion(y_pred, y_train)
    predicted = torch.max(y_pred.data,1)[1] # Add up the number of correct predictions, Indexed off the first point.
    batch_corr = (predicted == y_train).sum() # how many we got correct from this batch.
    trn_corr += batch_corr

    # update our parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if b%600 == 0:
      print(f'epoch: {i} batch: {b} loss: {loss.item()}')

  train_loss.append(loss.detach().numpy())
  train_correct.append(trn_corr)


  # TEST
  with torch.no_grad():
    for(b, (X_test, y_test)) in enumerate(test_loader):
      y_val = model(X_test)
      predicted = torch.max(y_val.data,1)[1]
      tst_corr += (predicted == y_test).sum()

  loss = criterion(y_val, y_test)
  test_loss.append(loss.detach().numpy())
  test_correct.append(tst_corr)

current_time = time.time()
total_time = current_time - start_time
print(f"Training time : {total_time/60} minutes")

train_correct

# Graph the loss at epoch
plt.plot(train_loss, label = "Train Loss")
plt.plot(test_loss, label = "Test Loss")
plt.title("Loss at Epoch")
plt.legend()

# Graph the accuracy at the end of each epoch
plt.plot([t/600 for t in train_correct], label = "Training accuracy")
plt.plot([t/100 for t in test_correct], label = "Testing  accuracy")
plt.title("Accuracy at Epoch")
plt.legend()
plt.show()

# accuracy for this model
print(f'Test accuracy: {test_correct[-1].item()*100/10000:.3f}%')

plt.imshow(test_data[4143][0][0])

model.eval()
with torch.no_grad():
  new_prediction = model(test_data[4143][0][0].view(1,1,28,28))

print(new_prediction.argmax())

# Saving the model as cnn_model.pt for MNIST
torch.save(model.state_dict(), "cnn_model.pt")

# Load the model
model1 = CNNModel()
model1.load_state_dict(torch.load('cnn_model.pt'))  # for using weight s

model2 = torch.load("cnn_model.pt")

