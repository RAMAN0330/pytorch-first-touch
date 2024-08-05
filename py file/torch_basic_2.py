import torch
import torch.nn as nn
import torch.nn.functional as f

# Create a Model class that inherits nn.Module
class Model(nn.Module):
  def __init__(self, in_features=4, h1 = 8, h2 = 9, out_features = 3):
    super().__init__()  # Initialize the parent class
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1,h2)
    self.out = nn.Linear(h2,out_features)

  def forward(self,x):
    x = f.relu(self.fc1(x))
    x = f.relu(self.fc2(x))
    x = self.out(x)

    return x

# Pick a random seed
torch.manual_seed(41)
model = Model()

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib  inline

data = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')

data.head()

print(data["variety"].value_counts())

data["variety"] = data["variety"].map({"Setosa":0.,"Versicolor":1,"Virginica":2})

# train test split set x,y and convert them into numpy array
x = data.drop("variety",axis=1).values
y = data["variety"].values

# Spliting data into train and test
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=41)

# convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# convert y features to long tensor
y_train  = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# checking dimension of each split
print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)

# set the criterion of model to measure error, how far off the prediction are from acutal result
criterion = nn.CrossEntropyLoss()
# Choose Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.parameters

# Train our Model
epochs = 100
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train)
  # Measure the loss/error, gonna be high at first
  loss  = criterion(y_pred,y_train) # predicted
  # Keep Track of Loss
  losses.append(loss.detach().numpy())
  # Every 10 epochs print epoch , loss
  if i % 10 == 00:
    print("Epochs : ", epochs, " loss : ",loss)

  # Do some backpropogation: take the error rate of forward progation and feed it backward to finetune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

plt.plot(range(epochs),losses)
plt.ylabel("Loss")
plt.xlabel("Epochs")

# Evaluate Model on Test Dataset-validate model on test dataset
with torch.no_grad():   # Basically turn off backpropogation
  y_eval = model.forward(X_test)  # dataset will be from Testing Sets
  loss = criterion(y_eval,y_test) # finding loss and error

loss

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)
    # Will tell us what type of flower class our network think it is
    print(f'{i+1:2}. {str(torch.argmax(y_val)):38} {y_test[i]}')

    if torch.argmax(y_val) == y_test[i]:
      correct +=1
print()
print("Number of corrected predicted is", correct)

# Save our Model
torch.save(model.state_dict(), "iris_model.pt")

# load the Model
new_model = Model()
new_model.load_state_dict(torch.load("iris_model.pt"))

new_model.eval()

