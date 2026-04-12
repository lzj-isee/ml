import torch

class Conv:
    def __init__(self, cin: int, cout: int, k: int) -> None:
        self.weight = torch.rand(cout, cin, k, k)
        self.bias = torch.rand(cout)
        self.k = k
        self.cin = cin
        self.cout = cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, cin, hin, win, = x.shape
        x = torch.nn.functional.unfold(x, kernel_size = self.k) # [b, cin * k * k, hout * wout]
        x = x.transpose(1, 2) # [b, hout * wout, cin * k * k]
        w = self.weight.view(self.cout, self.cin * self.k * self.k).transpose(0, 1)
        o = x @ w + self.bias.view(1, 1, self.cout) # [b, hout * wout, cout]
        hout = (hin - self.k + 2 * 0) // 1 + 1
        wout = (win - self.k + 2 * 0) // 1 + 1
        o = o.transpose(1, 2).view(b, self.cout, hout, wout)
        return o


def test_conv():
    torch.manual_seed(42)
    
    # 创建输入（确定性序列）
    b, cin, hin, win = 2, 3, 8, 8
    k = 3
    cout = 4
    x = torch.arange(b * cin * hin * win).float().view(b, cin, hin, win)
    
    # 自定义 Conv
    conv = Conv(cin, cout, k)
    conv.weight = torch.arange(cout * cin * k * k).float().view(cout, cin, k, k)
    conv.bias = torch.zeros(cout)
    
    # PyTorch Conv2d 作为参考
    ref = torch.nn.Conv2d(cin, cout, k, bias=True)
    ref.weight.data = conv.weight.clone()
    ref.bias.data = conv.bias.clone()
    
    # 对比输出
    out_custom = conv.forward(x)
    out_ref = ref(x)
    
    print(f"custom output shape: {out_custom.shape}")
    print(f"ref output shape: {out_ref.shape}")
    print(f"max diff: {(out_custom - out_ref).abs().max().item()}")
    print(f"allclose: {torch.allclose(out_custom, out_ref)}")


if __name__ == "__main__":
    test_conv()
    




