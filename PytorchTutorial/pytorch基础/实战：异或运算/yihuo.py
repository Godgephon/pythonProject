import torch
import torch.nn as nn

# 构造输入和期望输出
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# 数据类型转换
x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

# 神经网络
net = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    nn.Sigmoid()
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()
# 训练神经网络
for epoch in range(5000):
    out = net(x_tensor)
    loss = loss_func(out, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'迭代次数:{epoch}')
        print(f'误差:{loss}')


out = net(x_tensor)
print(f'out:{out.data}')


torch.save(net, 'net.pkl')
