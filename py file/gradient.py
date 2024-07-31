import torch

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

print(device)

x = torch.ones(3,2)
print(x)

x = torch.zeros(3,2)
print(x)

x = torch.rand(3,2)
print(x)

x = torch.empty(3,2)
print(x)

y = torch.zeros_like(x)
print(y)

x = torch.linspace(0,10,5)
print(x)

x.size()

x = torch.rand(3,2)
print(x.size())

x.view(2,3)

print(x[:,1])

"""x[1,1].item()"""

print(x[1,1].item())

x = torch.rand(3,2)
y = torch.rand(3,2)

print(x+y)

print(x*y)

print(x.matmul(y.T))

import numpy as np

x = np.array([[2,3,4],[5,6,7]])
print(x) # Numpy array

t = torch.from_numpy(x)
print(t) # converted to tensor

torch.cuda.get_device_name(0)

torch.rand(3,2)*torch.rand(3,2)

x = torch.rand(3,2,device = "cuda")

print(x)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# for i in range(1000):
#   x = torch.randn(1000,1000)
#   y = torch.randn(1000,1000)
#   z = x*y

## Auto Differentiation
x = torch.ones(3,2, requires_grad = True)
print(x)

y = x+5

print(y)

z = y*y + 3
print(z)

t = torch.sum(z)
print(t)

t.backward()
x.grad

w = torch.rand(1,4, device = "cuda").normal_(mean=0,std=1)
print(w)

b = torch.ones(size = (1,4))
print(b)

inp_mat = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(inp_mat)

