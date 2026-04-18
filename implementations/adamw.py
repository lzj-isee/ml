import torch

class AdamW:
    def __init__(self):
        self.size = 1024
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.var1 = torch.zeros((self.size, ))
        self.var2 = torch.zeros((self.size, ))
        self.weight_decay = 0.1
        self.w = torch.rand((self.size, ))
        self.cnt = 0
        self.lr = 0.001

    def step(self, grad: torch.Tensor) -> None:
        self.cnt += 1
        self.var1 = self.beta1 * self.var1 + (1 - self.beta1) * grad
        self.var2 = self.beta2 * self.var2 + (1 - self.beta2) * grad.pow(2)
        g = self.var1 / (1 - self.beta1 ** self.cnt)
        m = self.var2 / (1 - self.beta2 ** self.cnt)
        self.w = self.w - self.lr * (g * torch.rsqrt(m + 1e-06) + self.weight_decay * self.w)