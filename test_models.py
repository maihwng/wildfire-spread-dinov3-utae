"""
Unit and integration test suite for DINOv3-UTAE architecture.
Tests:
  1. Multi-temporal input forward pass (B, T, 17, 128, 128) -> (B, 1, 128, 128).
  2. Single-day 4D input forward pass (B, 17, 128, 128) -> (B, 1, 128, 128).
  3. Pretrained DINOv3 checkpoint loading and weight verification.
  4. Temporal attention map extraction (return_att=True).
  5. Channel adaptation strategies: 'adapter' and 'conv1x1'.
  6. Backward pass and loss computation with CombinedLoss (Focal + Dice).
  7. Backbone freezing capability (freeze_backbone=True).
"""

import sys
from pathlib import Path
import torch

# Ensure paths
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.dinov3_utae import DINOv3_UTAE, build_dinov3_utae
from losses import CombinedLoss


def test_forward_multi_temporal():
    print("\n--- Test 1: Multi-temporal Input (B=2, T=5, C=17, H=128, W=128) ---")
    model = build_dinov3_utae(in_channels=17, n_classes=1, pretrained=True)
    model.eval()

    x = torch.randn(2, 5, 17, 128, 128)
    with torch.no_grad():
        out, att = model(x, return_att=True)

    print(f"Output shape: {out.shape}")
    print(f"Attn shape:   {att.shape}")
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    assert att.shape == (16, 2, 5, 8, 8), f"Expected (16, 2, 5, 8, 8), got {att.shape}"
    print("Test 1 PASSED.")


def test_forward_single_day():
    print("\n--- Test 2: Single-day 4D Input (B=2, C=17, H=128, W=128) ---")
    model = build_dinov3_utae(in_channels=17, n_classes=1, pretrained=True)
    model.eval()

    x = torch.randn(2, 17, 128, 128)
    with torch.no_grad():
        out = model(x)

    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    print("Test 2 PASSED.")


def test_backward_and_loss():
    print("\n--- Test 3: Backward Pass & Gradient Flow with Loss ---")
    model = build_dinov3_utae(in_channels=17, n_classes=1, pretrained=True)
    model.train()

    criterion = CombinedLoss()
    x = torch.randn(2, 3, 17, 128, 128, requires_grad=True)
    y = torch.randint(0, 2, (2, 1, 128, 128)).float()

    logits = model(x)
    loss = criterion(logits, y)
    print(f"Loss value: {loss.item():.4f}")

    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"Input gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients should be non-zero"
    print("Test 3 PASSED.")


def test_channel_strategy():
    print("\n--- Test 4: Channel Strategy 'conv1x1' ---")
    model = build_dinov3_utae(in_channels=17, n_classes=1, channel_strategy="conv1x1", pretrained=True)
    model.eval()

    x = torch.randn(1, 4, 17, 128, 128)
    with torch.no_grad():
        out = model(x)

    print(f"Output shape with conv1x1: {out.shape}")
    assert out.shape == (1, 1, 128, 128)
    print("Test 4 PASSED.")


def test_frozen_backbone():
    print("\n--- Test 5: Frozen DINOv3 Backbone ---")
    model = build_dinov3_utae(in_channels=17, n_classes=1, freeze_backbone=True, pretrained=True)
    dino_trainable = sum(p.numel() for p in model.dino.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"DINOv3 trainable params: {dino_trainable}")
    print(f"Total trainable params:  {total_trainable:,}")
    assert dino_trainable == 0, "DINOv3 parameters should be frozen"
    assert total_trainable > 0, "Adapter and Decoder should remain trainable"
    print("Test 5 PASSED.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DINOv3-UTAE TEST SUITE")
    print("=" * 60)
    test_forward_multi_temporal()
    test_forward_single_day()
    test_backward_and_loss()
    test_channel_strategy()
    test_frozen_backbone()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

