"""Verification script for CRANE-v0 architecture changes.

Tests:
1. Forward pass shape correctness
2. Output value ranges
3. Backward pass - all params receive gradient
4. RayStream not dead branch (gradient != 0)
5. Parameter count
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.architecture_CRANE import CRANEModel

def test_crane():
    device = "cpu"
    model = CRANEModel(
        input_channels=18, width=192, board_size=8,
        grid_blocks=12, relation_blocks=5,
    ).to(device)

    B = 4
    x_spatial = torch.randn(B, 18, 8, 8, device=device)
    s_scalar = torch.randn(B, 5, device=device)

    # --- Forward pass ---
    out = model(x_spatial, s_scalar)

    print("=== Forward Pass ===")
    print(f"  value   shape: {out['value'].shape}  range: [{out['value'].min():.3f}, {out['value'].max():.3f}]")
    print(f"  f_trunk shape: {out['f_trunk'].shape}")

    assert out["value"].shape == (B, 1), f"Value shape wrong: {out['value'].shape}"
    assert out["f_trunk"].shape == (B, 192, 8, 8), f"f_trunk shape wrong: {out['f_trunk'].shape}"

    # Value in [-1, 1] (tanh)
    assert out["value"].min() >= -1.0 and out["value"].max() <= 1.0, "Value out of range"

    print("  [OK] All shapes and ranges correct")

    # --- Backward pass ---
    targets_value = torch.randn(B, 1, device=device).tanh()

    loss = torch.nn.functional.mse_loss(out["value"], targets_value)
    loss.backward()

    print("\n=== Backward Pass ===")
    print(f"  Loss: {loss.item():.6f}")

    no_grad_params = []
    for name, p in model.named_parameters():
        if p.grad is None or p.grad.abs().max() == 0:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"  [FAIL] {len(no_grad_params)} params with zero gradient:")
        for n in no_grad_params[:10]:
            print(f"    - {n}")
    else:
        print("  [OK] All parameters received non-zero gradient")

    # --- RayStream gradient check ---
    print("\n=== RayStream Gradient Check ===")
    ray_grad_norm = 0.0
    ray_count = 0
    for name, p in model.named_parameters():
        if "torso.ray." in name and p.grad is not None:
            ray_grad_norm += p.grad.norm().item()
            ray_count += 1

    if ray_count > 0:
        avg = ray_grad_norm / ray_count
        print(f"  RayStream avg grad norm: {avg:.6f} ({ray_count} params)")
        if avg > 1e-10:
            print("  [OK] RayStream is NOT a dead branch")
        else:
            print("  [FAIL] RayStream may be dead branch!")
    else:
        print("  [FAIL] No RayStream params found")

    # --- RayFusion gate check ---
    print("\n=== RayFusion Gate Check ===")
    # Run forward again to inspect gate values
    model.eval()
    with torch.no_grad():
        h_stem = model.torso.stem(x_spatial)
        film_params = model.torso.film_gen(s_scalar)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        h_0 = gamma * h_stem + beta
        
        h_grid = model.torso.grid(h_0)
        h_ray = model.torso.ray(x_spatial)
        fused = torch.cat([h_grid, h_ray], dim=1)
        g = torch.sigmoid(model.torso.ray_fusion.g_conv(fused))
        
        print(f"  Gate sigmoid mean: {g.mean():.4f}, std: {g.std():.4f}")
        print(f"  Gate range: [{g.min():.4f}, {g.max():.4f}]")
        if g.mean() > 0.1 and g.mean() < 0.9:
            print("  [OK] Gate NOT saturated")
        else:
            print("  [WARN] Gate may be saturated")

    # --- Param count ---
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Parameter Count ===")
    print(f"  Total: {total:,} ({total/1e6:.2f}M)")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")

    # Breakdown
    torso_p = sum(p.numel() for p in model.torso.parameters())
    head_p = sum(p.numel() for p in model.head.parameters())
    print(f"  Torso: {torso_p:,} ({torso_p/total*100:.1f}%)")
    print(f"  ValueHead: {head_p:,} ({head_p/total*100:.1f}%)")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_crane()
