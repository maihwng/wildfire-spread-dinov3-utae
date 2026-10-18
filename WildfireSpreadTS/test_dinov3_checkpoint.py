from pathlib import Path
import torch

repo_dir = Path(r"D:\WildfirePilot\dinov3")
weights_path = Path(
    r"D:\WildfirePilot\weights\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

if not repo_dir.is_dir():
    raise FileNotFoundError(f"Không tìm thấy mã nguồn: {repo_dir}")

if not weights_path.is_file():
    raise FileNotFoundError(f"Không tìm thấy checkpoint: {weights_path}")

print("Checkpoint size (MB):", round(weights_path.stat().st_size / 1024**2, 2))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

model = torch.hub.load(
    str(repo_dir),
    "dinov3_vits16",
    source="local",
    weights=str(weights_path),
)

model = model.to(device).eval()

x = torch.randn(1, 3, 288, 192, device=device)

with torch.inference_mode():
    features = model.forward_features(x)
    tokens = features["x_norm_patchtokens"]

print("Patch tokens shape:", tuple(tokens.shape))
print("DINOv3 pretrained checkpoint loaded successfully.")