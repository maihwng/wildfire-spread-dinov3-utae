from pathlib import Path

import torch
import torch.nn.functional as F

from src.dataloader.FireSpreadDataset import FireSpreadDataset
from src.models.utae_paps_models.utae import UTAE


# --------------------------------------------------
# 1. Đường dẫn
# --------------------------------------------------

DINO_REPO = Path(r"D:\WildfirePilot\dinov3")

DINO_WEIGHTS = Path(
    r"D:\WildfirePilot\weights"
    r"\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

if not DINO_REPO.is_dir():
    raise FileNotFoundError(f"Không tìm thấy {DINO_REPO}")

if not DINO_WEIGHTS.is_file():
    raise FileNotFoundError(f"Không tìm thấy {DINO_WEIGHTS}")


# --------------------------------------------------
# 2. Đọc dataset pilot thật
# --------------------------------------------------

dataset = FireSpreadDataset(
    data_dir="data/raw",
    included_fire_years=[2021],
    n_leading_observations=5,
    crop_side_length=128,
    load_from_hdf5=False,
    is_train=False,
    remove_duplicate_features=False,
    stats_years=[2020, 2021],
)

print("Tổng số mẫu:", len(dataset))

# Sample 2 có 28 pixel cháy theo kết quả kiểm tra trước.
sample_idx = 2
x, y = dataset[sample_idx]

print("Input:", tuple(x.shape))
print("Target:", tuple(y.shape))
print("Số pixel cháy trong nhãn:", int((y > 0).sum()))

# Các ngày đầu vào: 17/07 đến 21/07/2021.
# Ngày mục tiêu: 22/07/2021.
positions = torch.tensor(
    [[198, 199, 200, 201, 202]],
    dtype=torch.long,
)


# --------------------------------------------------
# 3. Khởi tạo DINOv3-U-TAE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = UTAE(
    input_dim=40,
    encoder_widths=[64, 64, 64, 128],
    decoder_widths=[32, 32, 64, 128],
    out_conv=[32, 1],
    str_conv_k=4,
    str_conv_s=2,
    str_conv_p=1,
    agg_mode="att_group",
    encoder_norm="group",
    n_head=16,
    d_model=256,
    d_k=4,
    encoder=False,
    return_maps=False,
    pad_value=0,
    padding_mode="reflect",
    spatial_backbone="dinov3",
    dinov3_repo_dir=str(DINO_REPO),
    dinov3_weights_path=str(DINO_WEIGHTS),
).to(device)

model.eval()


# --------------------------------------------------
# 4. Inference với dữ liệu thật
# --------------------------------------------------

x = x.unsqueeze(0).to(device)
y = y.unsqueeze(0).unsqueeze(1).float().to(device)
positions = positions.to(device)

with torch.inference_mode():
    logits = model(
        x,
        batch_positions=positions,
        return_att=False,
    )

    loss = F.binary_cross_entropy_with_logits(
        logits, y
    )

    probabilities = torch.sigmoid(logits)
    predicted_mask = probabilities >= 0.5

print("\n===== KẾT QUẢ REAL DATA =====")
print("Input shape:", tuple(x.shape))
print("Target shape:", tuple(y.shape))
print("Output shape:", tuple(logits.shape))
print("Loss:", loss.item())
print("Output finite:", torch.isfinite(logits).all().item())
print("Số pixel cháy thực tế:", int(y.sum().item()))
print("Số pixel mô hình dự đoán cháy:", int(predicted_mask.sum().item()))

if not torch.isfinite(logits).all():
    raise RuntimeError("Output có NaN hoặc Inf.")

if logits.shape != y.shape:
    raise RuntimeError(
        f"Shape dự đoán {tuple(logits.shape)} "
        f"không khớp nhãn {tuple(y.shape)}."
    )

print("\nPASS: Mô hình chạy được trên mẫu pilot thật.")