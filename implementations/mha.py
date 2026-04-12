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
    
class ROPE:
    def __init__(self, max_len: int, head_dim: int, freq_base: int = 10_000_000):
        self.max_len = max_len
        self.head_dim = head_dim
        self.freq_base = freq_base

    def gen_rope_embs(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(self.max_len)
        dim = torch.concat([torch.arange(self.head_dim // 2), torch.arange(self.head_dim // 2)])
        freq = torch.reciprocal(self.freq_base ** (2 * dim / self.head_dim))
        cos = torch.cos(pos.view(-1, 1) * freq.view(1, -1))
        sim = torch.sin(pos.view(-1, 1) * freq.view(1, -1))
        return cos, sim

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        sx = torch.concat([-x[..., -self.head_dim // 2:], x[..., :self.head_dim // 2]])
        seq_len = x.shape[2]
        cos = cos[:seq_len].view(1, 1, seq_len, self.head_dim)
        sin = sin[:seq_len].view(1, 1, seq_len, self.head_dim)
        o = cos * x + sin * sx
        return o
    
class Attention:
    def __init__(self, hidden_dim: int, num_q: int, num_kv: int, head_dim: int, max_len: int):
        self.q_linear = torch.nn.Linear(hidden_dim, num_q * head_dim, bias = False)
        self.k_linear = torch.nn.Linear(hidden_dim, num_kv * head_dim, bias = False)
        self.v_linear = torch.nn.Linear(hidden_dim, num_kv * head_dim, bias = False)
        self.o_linear = torch.nn.Linear(num_q * head_dim, hidden_dim, bias = False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.num_q = num_q
        self.num_kv = num_kv
        self.head_dim = head_dim
        self.rope = ROPE(max_len, head_dim)
        self.cos, self.sin = self.rope.gen_rope_embs()

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
        q = self.rope.apply_rope(q, self.cos, self.sin)
        k = self.rope.apply_rope(k, self.cos, self.sin)

        qk = q @ k.transpose(-1, -2)
        mask = torch.where(torch.tril(torch.ones(seq)) > 0, 0, torch.finfo(torch.float32).min)
        m = torch.softmax(qk + mask, dim = -1) # b, m, s, s
        o = m @ v # b, m, s, d
        o = self.o_linear(o.transpose(1, 2).view(batch, seq, -1))
        return o # b, s, h
    
class FNN:
    def __init__(self, hidden_dim: int, inter_dim: int) -> None:
        self.up = torch.nn.Linear(hidden_dim, inter_dim, bias = False)
        self.down = torch.nn.Linear(inter_dim, hidden_dim, bias = False)
        self.gate = torch.nn.Linear(hidden_dim, inter_dim, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.up.forward(x) # [b, s, inter]
        x2 = self.gate.forward(x) # [b, s, inter]
        x3 = torch.nn.functional.silu(x2) # [b, s, inter]
        out = self.down.forward(x3 * x1)
        return out

class Decoder:
    def __init__(self, hidden_dim: int, num_q: int, num_kv: int, head_dim: int, inter_dim: int, max_len: int):
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_q, num_kv, head_dim, max_len)
        self.fnn_norm = RMSNorm(hidden_dim)
        self.fnn = FNN(hidden_dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.attn_norm.forward(x)
        x1 = self.attn.forward(x1)
        x = x + x1
        x1 = self.fnn_norm.forward(x)
        x1 = self.fnn.forward(x1)
        x = x + x1
        return x