import torch


class RMSNorm:
    def __init__(self, hidden_d: int) -> None:
        self.eps = 1e-8
        self.w = torch.nn.Parameter(torch.ones((hidden_d, )))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_d]
        return x * torch.rsqrt(self.eps + x.pow(2).mean(dim = -1, keepdim = True)) * self.w
    
class ROPE:
    def __init__(self, half_dim: int, max_len: int) -> None:
        self.base = 10_000_000
        self.freq = (self.base ** (- torch.cat([torch.arange(half_dim), torch.arange(half_dim)]) / half_dim))
        self.cos = torch.cos(torch.arange(max_len).view(-1, 1) * self.freq.view(1, -1)) # [i, d]
        self.sin = torch.sin(torch.arange(max_len).view(-1, 1) * self.freq.view(1, -1)) # [i, d]
    
    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, num_head, s, head_dim]
        b, num, s, d = x.shape
        cos = self.cos[:s]
        sin = self.sin[:s]
        o = cos.view(1, 1, s, d)* x + sin.view(1, 1, s, d) * torch.cat([-x[..., -d//2:], x[..., :d//2]])
        return o

class MLP:
    def __init__(self, hidden_d: int, moe_d: int) -> None:
        self.up = torch.nn.Linear(hidden_d, moe_d, bias = False)
        self.gate = torch.nn.Linear(hidden_d, moe_d, bias = False)
        self.down = torch.nn.Linear(moe_d, hidden_d, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., h]
        y1 = self.up.forward(x) # [..., moe_d]
        y2 = torch.nn.functional.silu(self.gate.forward(x))
        y3 = self.down(y1 * y2)
        return y3 # [..., h]
    
class Router:
    def __init__(self, hidden_d: int, num_experts: int, act_experts: int) -> None:
        self.router = torch.nn.Linear(hidden_d, num_experts, bias = False)
        self.num_experts = num_experts
        self.act_experts = act_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: # indices, weights
        # x[..., h]
        x1 = self.router.forward(x).softmax(dim = -1) # [..., num_experts]
        indices, weights = torch.topk(x1, k = self.act_experts, dim = -1)
        weights = weights / weights.sum(dim = -1, keepdim = True)
        return indices, weights # [..., act_experts]


class MOE:
    def __init__(self, hidden_d: int, moe_d: int, num_experts: int, act_experts: int) -> None:
        self.experts = [MLP(hidden_d, moe_d) for _ in range(num_experts)]
        self.router = Router(hidden_d, num_experts, act_experts)
        self.num_experts = num_experts
        self.act_experts = act_experts
        self.hidden_d = hidden_d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[..., h]
        ori_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1]) # [n, h]
        indices, weights = self.router.forward(x) # [n, act_experts]
        o = torch.zeros((x.shape[0], self.act_experts, self.hidden_d)) # [n, act_experts, h]
        for i in range(self.num_experts):
            token_idx, pos_idx = torch.where(indices == i) # [m, ], [m, ]
            if token_idx.numel() == 0:
                continue
            states = self.experts[i].forward(x[token_idx]) # [m, h]
            o[(token_idx, pos_idx)] = states
        o = (weights.unsqueeze(-1) * o).sum(dim = 1, keepdim = False) # [n ,h]
        return o.view(*ori_shape, self.hidden_d)

class Attention:
    def __init__(self, hidden_d: int, head_d: int, num_q: int, num_kv: int) -> None:
        self.q_proj = torch.nn.Linear(hidden_d, head_d * num_q, bias = False)
        self.k_proj = torch.nn.Linear(hidden_d, head_d * num_kv, bias = False)
        self.v_proj = torch.nn.Linear(hidden_d, head_d * num_kv, bias = False)
        self.q_norm = RMSNorm(hidden_d)
        self.k_norm = RMSNorm(hidden_d)
        self.o_proj = torch.nn.Linear(head_d * num_q, hidden_d, bias = False)
        self.head_d = head_d
        self.num_q = num_q
        self.num_kv = num_kv
        self.rope = ROPE(half_dim = head_d // 2, max_len = 32_000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, s, hidden_d]
        b, s, hidden_d = x.shape
        q = self.q_proj.forward(x) # [b, s, head_d * num_q]
        k = self.k_proj.forward(x) # [b, s, head_d * num_k]
        v = self.v_proj.forward(x) # [b, s, head_d * num_k]
        q = q.view(b, s, self.num_q, self.head_d)
        k = k.view(b, s, self.num_kv, self.head_d)
        v = v.view(b, s, self.num_kv, self.head_d)
        q = self.q_norm.forward(q) # [b, s, num_q, head_d]
        k = self.k_norm.forward(k) # [b, s, num_k, head_d]
        q = q.transpose(1, 2) # [b, num_q, s, head_d]
        k = k.transpose(1, 2) # [b, num_k, s, head_d]
        v = v.transpose(1, 2) # [b, num_k, s, head_d]
        q = self.rope.apply_rope(q)
        k = self.rope.apply_rope(k)
        k = k.repeat_interleave(self.num_q // self.num_kv, dim = 1) # [b, num_q, s, head_d]
        v = v.repeat_interleave(self.num_q // self.num_kv, dim = 1) # [b, num_q, s, head_d]
        logits = q @ k.transpose(-1, -2) # [b, num_q, s, s]
        mask = torch.where(torch.ones((s, s), dtype = torch.bool).tril(), 0, torch.finfo(torch.bfloat16).min)
        qk_attn = (logits * mask.view(1, 1, s, s)).softmax(dim = -1) # [b, num_q, s, s]
        avg_v = qk_attn @ v # [b, num_q, s, head_d]
        avg_v = avg_v.transpose(1, 2).view(b, s, self.num_q * self.head_d) # [b, s, num_q * head_d]
        o = self.o_proj.forward(avg_v) # [b, s, hidden_d]
        return o
    
class Decoder:
    def __init__(self, hidden_d: int, moe_d: int, head_d: int, num_q: int, num_kv: int, num_experts: int, act_experts: int) -> None:
        self.attn = Attention(hidden_d, head_d, num_q, num_kv)
        self.moe = MOE(hidden_d, moe_d, num_experts, act_experts)
        self.norm_attn = RMSNorm(hidden_d)
        self.norm_moe = RMSNorm(hidden_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.norm_attn.forward(x)
        x1 = self.attn.forward(x1)
        x = x + x1
        x1 = self.norm_moe.forward(x)
        x1 = self.moe.forward(x1)
        x = x + x1
        return x

