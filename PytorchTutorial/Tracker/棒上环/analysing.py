import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

data = pd.read_csv('数据.csv', encoding='utf-8', delimiter='\t')
data = data.dropna()
x = data['t']
y = data['x']
x = np.array(x)
y = np.array(y)
matplotlib.rc('font', family='SimHei')
matplotlib.rc('axes', unicode_minus=False)
plt.figure(figsize=(9, 9))
plt.scatter(x, y, color='black', linewidths=2.5)
plt.grid()
plt.xlabel('时间(s)', fontsize=15)
plt.ylabel('速度(m/s)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)
x_tensor = torch.unsqueeze(x_tensor, dim=1)
y_tensor = torch.unsqueeze(y_tensor, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        prediction = self.predict(x)
        return prediction


net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()
train_times = 5000

for epoch in range(train_times):
    out = net(x_tensor)
    loss = loss_func(out, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(x, out.data.numpy(), color='red', label='拟合曲线')
plt.savefig('output.jpg', dpi=100)
