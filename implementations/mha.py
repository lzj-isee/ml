import torch

# according to transformers.models.qwen3.modeling_qwen3

class RMSNorm:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.scale = torch.nn.Parameter(data = torch.ones(hidden_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, ..., d]"""
        _dtype = x.dtype
        factor = (x.to(torch.float32).pow(2).mean(-1, keepdim = True) + 1e-6).rsqrt()
        return self.scale * x * factor.to(dtype = _dtype)
    
class Attention:
    def __init__(self, hidden_dim: int, num_q: int, num_kv: int, head_dim: int):
        self.q_linear = torch.nn.Linear(hidden_dim, num_q * head_dim, bias = False)
        self.k_linear = torch.nn.Linear(hidden_dim, num_kv * head_dim, bias = False)
        self.v_linear = torch.nn.Linear(hidden_dim, num_kv * head_dim, bias = False)
        self.o_linear = torch.nn.Linear(num_q * head_dim, hidden_dim, bias = False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.num_q = num_q
        self.num_kv = num_kv
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """NOTE: omit rope now, practice next time"""
        # x: [b, s, h]
        batch = x.shape[0]
        seq = x.shape[1]
        q: torch.Tensor = self.q_linear(x).view((*x.shape[:-1], self.num_q, self.head_dim)) # b, s, m, d
        k: torch.Tensor = self.k_linear(x).view((*x.shape[:-1], self.num_kv, self.head_dim)) # b, s, n, d
        v: torch.Tensor = self.v_linear(x).view((*x.shape[:-1], self.num_kv, self.head_dim)) # b, s, n, d
        q = self.q_norm.forward(q)  # b, s, m, d
        k = self.k_norm.forward(k).expand_as(q).transpose(1, 2)  # b, m, s, d
        q = q.transpose(1, 2) # b, m, s, d
        v = v.expand_as(q).transpose(1, 2) # b, m, s, d
        # omit rope right now

        qk = q @ k.transpose(-1, -2)
        mask = torch.where(torch.tril(torch.ones(seq)) > 0, 0, torch.finfo(torch.float32).min)
        m = torch.softmax(qk + mask, dim = -1) # b, m, s, s
        o = m @ v # b, m, s, d
        o = self.o_linear(o.transpose(1, 2).view(batch, seq, -1))
        return o # b, s, h
    
class FNN:
    def __init__(self) -> None:
        pass









        