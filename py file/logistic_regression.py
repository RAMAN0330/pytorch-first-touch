# Logistic Regression with PyTorch: Because Nothing Says Simple Like an Exponential Curve
#So, here we are, tackling the age-old problem of classifying data points into two distinct classes. Clearly, a straight line wasn't dramatic enough,so we opted for logistic regression—a method that bends and twists to fit our needs. And who better to handle this than PyTorch, our favorite deep learning library? Let's walk through this ordeal with as much enthusiasm as one can muster. 🎉
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

train_ds = MNIST(root = "data/", download = True, transform = transforms.ToTensor(), train = True)
test_ds = MNIST(root = "data/", download = True, transform = transforms.ToTensor(), train = False)

plt.imshow(train_ds[0][0][0], cmap = "gray")
plt.axis("off")

train_ds, val_ds = random_split(train_ds, [50000, 10000])
len(train_ds), len(val_ds)

# Making batches
train_loader = DataLoader(train_ds, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 128, shuffle = True)
test_loader = DataLoader(test_ds, batch_size = 128, shuffle = True)

# Model
input_size = 28*28
num_classes = 10
model = nn.Linear(input_size, num_classes)
list(model.parameters())

for images, labels in train_loader:
  images = images.reshape(-1,28*28)
  outputs = model(images)
  break

"""Model Definition with nn.Linear 💻:
The cornerstone of our logistic regression model is a linear layer. Think of nn.Linear as the intellectual equivalent of drawing a straight line, but with all the flair of matrix multiplication. In other words, it takes the input, applies a linear transformation, and spits out an output. It's the brain of our operation 🧠, but let's not pretend it's doing anything more than adding up weighted inputs and a bias term.
"""

class Logistic_regression_MNIST(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(input_size, num_classes)
  def forward(self, x):
    x = x.reshape(-1, 28*28)
    outputs = self.linear(x)
    return outputs

model = Logistic_regression_MNIST()
print(model.linear.weight.shape , model.linear.bias.shape)

for images , labels in train_loader:
  outputs = model(images)
  break

print(f"Output Shape : {outputs.shape}")
print(f"Sample Output : {outputs[:2].data}")

"""Softmax Function: Making Sure Everything Sums to One 🧮:
Next, we introduce F.softmax into the mix. Why, you ask? Because after all the complex calculations, we need to normalize our results. We want our outputs to look like probabilities, summing to one, rather than some arbitrary numbers. Softmax takes our linear output and graciously transforms it into something that resembles a probability distribution. It's like magic 🎩✨, but with more exponentials and less sleight of hand.
"""

# Applysoftmaxfor each output row
probs = F.softmax(outputs, dim =1 )
print(f"Sample Probabilities : {probs[:2].data}")
print(f"Sum of Probabilities : {probs[0].sum()}")

max_probs, preds = torch.max(probs, dim = 1)
print(f"Predictions : {preds}")
print(f"Max Probabilities : {max_probs}")

# Accuracy
def accuracy(outputs, label):
  _, preds = torch.max(outputs, dim = 1)
  return torch.tensor(torch.sum(preds == label).item() / len(preds))

#Can't be used as it's not differentiable and can't be valued in backproagation

# Cross-Entropy Loss: The Necessary Evil 😈:
#F.cross_entropy, our measure of how wrong our model is. This function is like a judgmental friend, constantly pointing out the flaws in our model's predictions. It calculates the difference between the predicted probabilities and the actual classes, providing a single value that tells us just how much our model needs to improve. In simpler terms, it's our guiding compass 🧭, leading us to the mythical land of lower loss.


def cross_entropy(outputs, labels):
  _, preds = torch.max(outputs, dim = 1)
  return F.cross_entropy(outputs, labels)

for _, labels in test_loader:
  print(cross_entropy(outputs, labels))
  break

"""Lower the loss, better the model"""
print(f"loss : {torch.exp(torch.tensor(-2.3186))}")

class Mnist_Logistic_regression(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    outputs = x.reshape(-1, 784) # Reshaping data for input_size
    outputs = self.linear(outputs)  # logistic model
    return outputs

  def training_step(self, batch):
    images, labels = batch
    outputs = self(images) # Generate predictions
    loss = F.cross_entropy(outputs, labels) # calculating loss
    return loss

  def validation_step(self, batch):
    images, labels = batch
    outputs = self(images)
    loss = F.cross_entropy(outputs, labels)
    acc = accuracy(outputs, labels)     # calculating accuracy
    return {"val_loss" : loss, "val_acc" : acc}

  def validation_epoch_end(self, outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean() # Combining loss
    batch_accs = [x["val_acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean() # cobining accuracy
    return {"val_loss" : epoch_loss.item(), "val_acc" : epoch_acc.item()}

  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["val_loss"], result["val_acc"]))


model = Mnist_Logistic_regression()

"""Optimizer Setup: The Art of Gradient Descent 🎨🖌️:
With the loss calculated, we need a way to adjust our model's parameters. Here comes the optimizer, with optim.SGD (Stochastic Gradient Descent) being the loyal workhorse 🐴 of the optimization world. It's simple, it's effective, and it has a learning rate that you'll never get right on the first try. This optimizer takes the calculated gradients and adjusts the weights, nudging our model ever so slightly toward better performance. 🚀
"""

def evaluate(model, val_loader):
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
  history = []
  optimizer = opt_func(model.parameters(), lr)
  for epoch in range(epochs):
    for batch in train_loader:
      loss = model.training_step(batch)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    result = evaluate(model, val_loader)
    model.epoch_end(epoch, result)
    history.append(result)
  return history

"""Backward Propagation: Letting the Gradients Do Their Thing 🌊:
To adjust the weights, we need to compute how much each weight contributed to the loss. This is where backpropagation steps in. It’s like a blame game for the weights—each one gets its fair share of guilt for the current prediction errors. Using loss.backward(), PyTorch automates this process, computing the gradients for us. It's efficient, it's automatic, and it makes us feel smart. 🤓

Optimizer Step: Taking a Step in the Right Direction 👣:
After all the gradient calculations, it's time to take a step. The optimizer.step() function is our gentle push towards better accuracy. It adjusts the model’s parameters based on the gradients, moving us closer to the optimal weights. Of course, this step is as good as our learning rate—too big and we overshoot, too small and we crawl. But hey, who doesn't love a little trial and error? 🎯
"""

# Evaluation o dataset before training
res = evaluate(model, val_loader)
res

# TRAIN THE MODEL
print("history 1")
history1 = fit(5, 0.001, model, train_loader, val_loader)
print("history 2")
history2 = fit(5, 0.001, model, train_loader, val_loader)
print("history 3")
history3 = fit(5, 0.001, model, train_loader, val_loader)

# plotting the loss and accuracy
plt.figure(figsize = (10,9))
history = history1 + history2 +history3
accuraciesv = [result["val_acc"] for result in history]
plt.plot(accuraciesv, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');

"""Testing on individual images"""

img, label = test_ds[1]
plt.imshow(img[0], cmap = "gray")
print('Label:', label, ', Predicted:', model(img.unsqueeze(0)).argmax().item())

# run on complete test dataset
def predict_image(img, label):
  x = img.unsqueeze(0)
  y = model(x)
  _, predicted = torch.max(y.data, dim = 1)
  print(f"Label : {label}, Predicted : {predicted}")

i = 0
for img, label in test_ds:
  if i <= 8:
    plt.subplot(3,3,i+1)
    predict_image(img, label)
    plt.imshow(img[0], cmap = "gray")
    plt.axis("off")
  else:
    break
  i += 1

# Evaluation on test
result = evaluate(model, test_loader)
result

"""Saving the Model State: Because Losing Progress is a Nightmare 🛟:
Finally, after all the hard work (and countless epochs of training), we save our model's state using torch.save. This function ensures that the next time we want to show off our model's performance, we won't have to retrain it from scratch. We save the state dictionary, which contains all the model's learned parameters. It's like saving a game 🎮; nobody wants to start over from level one.
"""

torch.save(model.state_dict(), "model-mnist-logistic-Weight-bias.pth")

torch.save(model, "model-mnist-logistic-complete.pth")

model.state_dict()

# for re-loading the model
model = Mnist_Logistic_regression()
model.load_state_dict(torch.load("model-mnist-logistic.pth"))  # to load model weight

model.parameters
