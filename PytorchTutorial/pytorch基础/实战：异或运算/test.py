from typing import Any

import torch
from torch import Tensor
from torch.jit import RecursiveScriptModule

x: list[list[int]] = [[0, 0]]
x_tensor: Tensor = torch.tensor(x, dtype=torch.float)
net: RecursiveScriptModule | RecursiveScriptModule | Any = torch.load('net.pkl')
out: object = net(x_tensor)
out_final: float = out.data
if out_final > 0.5:
    out_final = 1
else:
    out_final = 0
print(f'out={out_final}')
