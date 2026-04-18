import torch

def col2im(x: torch.Tensor, cin: int, hin: int, win: int, k: int) -> torch.Tensor:
    """
    x: [b, cin * k * k, L] where L = (hin - k + 1) * (win - k + 1)
    Returns: [b, cin, hin, win]
    """
    b = x.shape[0]
    indices = torch.arange(0, b * cin * hin * win).view(b, cin, hin, win).float()
    unfold_indices = torch.nn.functional.unfold(indices, kernel_size=k).long()
    out = torch.zeros((b * cin * hin * win,)).scatter_add_(dim=0, index=unfold_indices.view(-1), src=x.view(-1))
    return out.view(b, cin, hin, win)


def test_col2im():
    torch.manual_seed(42)

    b, cin, hin, win = 2, 3, 8, 8
    k = 3
    L = (hin - k + 1) * (win - k + 1)  # 6 * 6 = 36

    # 创建输入
    x = torch.arange(b * cin * k * k * L).float().view(b, cin * k * k, L)

    # 自定义 col2im
    out_custom = col2im(x, cin, hin, win, k)

    # PyTorch fold 作为参考
    out_ref = torch.nn.functional.fold(x, output_size=(hin, win), kernel_size=k)

    print(f"custom output shape: {out_custom.shape}")
    print(f"ref output shape: {out_ref.shape}")
    print(f"max diff: {(out_custom - out_ref).abs().max().item()}")
    print(f"allclose: {torch.allclose(out_custom, out_ref)}")


if __name__ == "__main__":
    test_col2im()