"""CPU contracts for the released modules; no dataset or full model is exercised."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from drpmnet import DRDBN, EACL, PMAF


def check_drdb():
    for mode in ('dual', 'spatial', 'context'):
        block = DRDBN(in_channels=32, mode=mode)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        output = block(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        if mode == 'dual':
            stats = block.get_routing_stats()
            assert abs(sum(stats.values()) - 1.0) < 1e-5
            block.reset_stats()
            assert block.count.item() == 0


def check_fusion():
    block = PMAF(in_channels_list=[32, 64, 128], out_channels=32)
    features = [torch.randn(2, c, s, s, requires_grad=True)
                for c, s in [(32, 16), (64, 8), (128, 4)]]
    expected = [(2, 32, s, s) for s in (16, 8, 4)]
    block.eval()
    with torch.no_grad():
        output = block(features)
    assert isinstance(output, list) and [tuple(x.shape) for x in output] == expected
    block.train()
    targets = [torch.rand(2, 1, s, s) for s in (16, 8, 4)]
    output, losses = block(features, targets)
    assert [tuple(x.shape) for x in output] == expected and len(losses) == 3
    loss = sum(x.square().mean() for x in output) + sum(losses)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in features)
    # The supplied implementation supports absent per-scale edge annotations.
    _, absent_losses = block([x.detach() for x in features], [None, None, None])
    assert len(absent_losses) == 3 and all(x.item() == 0 for x in absent_losses)
    assert all(x.requires_grad for x in absent_losses)


def check_eacl():
    block = EACL(in_channels=32, embed_dim=16)
    strong = torch.randn(2, 32, 16, 16, requires_grad=True)
    weak = torch.randn(2, 32, 16, 16, requires_grad=True)
    # More than 200 candidates exercises the source's sampling cap.
    edges = torch.ones(2, 1, 16, 16)
    loss = block(strong, weak, edges)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert strong.grad is not None and torch.isfinite(strong.grad).all()
    assert strong.grad.abs().sum().item() > 0
    assert weak.grad is None  # Weak target representation uses stop-gradient.
    for candidates in (0, 9):
        block.zero_grad(set_to_none=True)
        strong = torch.randn(2, 32, 8, 8, requires_grad=True)
        weak = torch.randn_like(strong, requires_grad=True)
        edges = torch.zeros(2, 1, 8, 8)
        edges.reshape(2, -1)[:, :candidates] = 1
        loss = block(strong, weak, edges)
        assert torch.isfinite(loss) and loss.item() == 0 and loss.requires_grad
        loss.backward()
        assert strong.grad is not None and torch.isfinite(strong.grad).all()
    # Also check the legacy learned-map fallback for basic numerical behavior.
    fallback = block(torch.randn(2, 32, 8, 8), torch.randn(2, 32, 8, 8))
    assert fallback.ndim == 0 and torch.isfinite(fallback)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.set_num_threads(2)
    check_drdb()
    check_fusion()
    check_eacl()
    print(f'PASS: standalone module checks (PyTorch {torch.__version__}, CPU).')
    print('Full-model training, paper accuracy, and deployment timing were not tested.')
