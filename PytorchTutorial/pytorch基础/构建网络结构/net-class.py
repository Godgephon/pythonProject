import torch.nn as nn


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output


net = myNet()
print(net)
