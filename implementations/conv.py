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
        w = self.weight.view(self.cout, self.cin * self.k * self.k).transpose(0, 1) # [cin * k * k, cout]
        o = x @ w + self.bias.view(1, 1, self.cout) # [b, hout * wout, cout]
        hout = (hin - self.k + 2 * 0) // 1 + 1
        wout = (win - self.k + 2 * 0) // 1 + 1
        o = o.transpose(1, 2).view(b, self.cout, hout, wout)
        return o
    
    def backward(self, dy: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        dy: [b, cout, hout, wout]
        x: [b, cin, hin, hout]
        weight: [cout, cin, k, k]
        bias: [cout, ]
        """
        k = self.k
        b, cout, hout, wout = dy.shape
        dy1 = dy.view(b, cout, hout * wout).transpose(1, 2) # [b, hout * wout, cout]
        x2 = torch.nn.functional.unfold(x, kernel_size = self.k).transpose(1, 2) # [b, hout * wout, cin * k * k]
        db = dy1.sum(dim = (0, 1))
        dw = dy1.view(b, hout * wout, cout, 1) * x2.view(b, hout * wout, 1, self.cin * self.k ** 2)
        dw = dw.sum(dim = (0, 1)).view(cout, self.cin, self.k, self.k)
        w2 = self.weight.view(cout, self.cin * self.k ** 2) # [cout, cin * k * k]
        dx = dy1 @ w2 # [b, hout * wout, cin * k * k]
        dx1 = torch.nn.functional.fold(dx.transpose(1, 2), output_size = (hout - 1 + k, wout - 1 + k), kernel_size = k)
        return dx1, dw, db

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
    ref.bias.data = conv.bias.clone() # type:ignore
    
    # 对比输出
    out_custom = conv.forward(x)
    out_ref = ref(x)
    
    print(f"custom output shape: {out_custom.shape}")
    print(f"ref output shape: {out_ref.shape}")
    print(f"max diff: {(out_custom - out_ref).abs().max().item()}")
    print(f"allclose: {torch.allclose(out_custom, out_ref)}")


def test_conv_backward():
    torch.manual_seed(42)
    
    # 创建输入
    b, cin, hin, win = 2, 3, 8, 8
    k = 3
    cout = 4
    x = torch.arange(b * cin * hin * win).float().view(b, cin, hin, win)
    x_ref = x.clone().requires_grad_(True)
    
    # 自定义 Conv
    conv = Conv(cin, cout, k)
    conv.weight = torch.arange(cout * cin * k * k).float().view(cout, cin, k, k)
    conv.bias = torch.zeros(cout)
    
    # PyTorch Conv2d 作为参考
    ref = torch.nn.Conv2d(cin, cout, k, bias=True)
    ref.weight.data = conv.weight.clone()
    ref.bias.data = conv.bias.clone() # type:ignore
    ref.weight.requires_grad_(True) # type:ignore
    ref.bias.requires_grad_(True) # type:ignore
    
    # forward
    out_custom = conv.forward(x)
    out_ref = ref(x_ref)
    
    # backward: 使用 sum of squares 作为 loss
    dy = torch.ones_like(out_custom)
    dx_custom, dw_custom, db_custom = conv.backward(dy, x)
    
    out_ref.backward(dy)
    dx_ref = x_ref.grad
    dw_ref = ref.weight.grad
    db_ref = ref.bias.grad # type:ignore
    
    # 对比梯度
    print("\n=== Backward Test ===")
    print(f"dx max diff: {(dx_custom - dx_ref).abs().max().item()}") # type:ignore
    print(f"dx allclose: {torch.allclose(dx_custom, dx_ref)}") # type:ignore
    print(f"dw max diff: {(dw_custom - dw_ref).abs().max().item()}") # type:ignore
    print(f"dw allclose: {torch.allclose(dw_custom, dw_ref)}") # type:ignore
    print(f"db max diff: {(db_custom - db_ref).abs().max().item()}") # type:ignore
    print(f"db allclose: {torch.allclose(db_custom, db_ref)}") # type:ignore


if __name__ == "__main__":
    test_conv()
    test_conv_backward()
    




