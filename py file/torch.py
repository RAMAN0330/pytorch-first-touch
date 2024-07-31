import torch

my_torch = torch.arange(10)
print(my_torch)

my_torch = my_torch.reshape(2,5)
print(my_torch)

torch_2 = my_torch.reshape(2,-1)
print(torch_2)

my_torch2 = torch.arange(10)
print(my_torch2)

my_torch2 = my_torch2.view(2,5)
print(my_torch2)

my_torch2 = my_torch2.clone()
print(my_torch2)

my_torch3 = torch.arange(10)
my_torch3 = my_torch3.view(5,-1)
print(my_torch3)

print(my_torch3[:,1:]) # for retaining the structure