import numpy as np


def linear_forward(x, W: np.ndarray, b):
    """
    x: b * d_in
    W: d_out * d_in
    b: d_out
    y: b * d_out
    """
    y = np.matmul(x, W.T) + b
    return y


def linear_backward(dy: np.ndarray, x, W: np.ndarray):
    """
    dy: b * d_out
    x: b * d_in
    W: d_out * d_in

    Returns:
        dx: b * d_in
        dW: d_out * d_in
        db: d_out
    """
    dx = dy @ W
    dW = np.einsum("bi,bj->ij", dy, x)
    db = np.sum(dy, axis=0)
    return dx, dW, db


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    np.random.seed(42)
    torch.manual_seed(42)

    # 设置参数
    batch_size = 4
    d_in = 6
    d_out = 3

    # 生成随机数据
    x_np = np.random.randn(batch_size, d_in).astype(np.float64)
    W_np = np.random.randn(d_out, d_in).astype(np.float64)
    b_np = np.random.randn(d_out).astype(np.float64)
    dy_np = np.random.randn(batch_size, d_out).astype(np.float64)

    # PyTorch 对照
    x_torch = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    dy_torch = torch.tensor(dy_np, dtype=torch.float64)

    # 前向传播
    linear = nn.Linear(d_in, d_out, dtype=torch.float64)
    linear.weight.data = torch.tensor(W_np, dtype=torch.float64)
    linear.bias.data = torch.tensor(b_np, dtype=torch.float64)

    y_torch: torch.Tensor = linear(x_torch)
    y_np = linear_forward(x_np, W_np, b_np)

    # 验证 forward
    print("=== Forward Test ===")
    print(f"y_np shape: {y_np.shape}, y_torch shape: {y_torch.shape}")
    print(f"Forward max diff: {np.max(np.abs(y_np - y_torch.detach().numpy()))}")
    print(f"Forward pass: {'✓ PASSED' if np.allclose(y_np, y_torch.detach().numpy()) else '✗ FAILED'}")

    # 反向传播
    dx_np, dW_np, db_np = linear_backward(dy_np, x_np, W_np)

    y_torch.backward(dy_torch)

    # 验证 backward
    print("\n=== Backward Test ===")
    print(f"dx max diff: {np.max(np.abs(dx_np - x_torch.grad.numpy()))}")
    print(f"dW max diff: {np.max(np.abs(dW_np - linear.weight.grad.numpy()))}")
    print(f"db max diff: {np.max(np.abs(db_np - linear.bias.grad.numpy()))}")

    dx_ok = np.allclose(dx_np, x_torch.grad.numpy())
    dW_ok = np.allclose(dW_np, linear.weight.grad.numpy())
    db_ok = np.allclose(db_np, linear.bias.grad.numpy())

    print(f"dx: {'✓ PASSED' if dx_ok else '✗ FAILED'}")
    print(f"dW: {'✓ PASSED' if dW_ok else '✗ FAILED'}")
    print(f"db: {'✓ PASSED' if db_ok else '✗ FAILED'}")
    print(f"\nAll tests: {'✓ PASSED' if (dx_ok and dW_ok and db_ok) else '✗ FAILED'}")