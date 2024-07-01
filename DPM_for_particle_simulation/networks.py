import torch
from torch import nn


class ModelNN(nn.Module):
    def __init__(self, n=40):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )
        self.tail = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128,128), nn.ReLU(),
                nn.Linear(128,2*2) # dimension * 2 (loc and scale)
            ) for _ in range(n)
        ])

    def forward(self, x, t):
        x = self.head(x)
        x = self.tail[t](x)
        return x


if __name__ == "__main__":
    input_1 = torch.randn(2)
    print(f"input_1={input_1}")

    f = ModelNN(n=10)
    output_1 = f(x=input_1, t=3)
    print(f"output_1={output_1}")