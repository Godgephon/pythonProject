import torch

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([[1, 2], [3, 4]])
print(f'x:{x},x的类型:{x.type()},x的尺寸:{x.size()}')
print(f'y:{y},y的类型:{y.type()},y的尺寸:{y.size()}')
