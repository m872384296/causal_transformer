import torch

class split_param(torch.nn.Module):
    def __init__(self, length, n_env):
        super().__init__()
        self.soft_split = torch.randn((length, n_env))
        self.soft_split = torch.nn.Parameter(self.soft_split)