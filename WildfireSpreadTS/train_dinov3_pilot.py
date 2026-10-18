from pathlib import Path
from datetime import date, timedelta

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.dataloader.FireSpreadDataset import FireSpreadDataset
from src.models.utae_paps_models.utae import UTAE


# =========================
# CONFIG
# =========================
DINO_REPO = Path(r"D:\WildfirePilot\dinov3")
DINO_WEIGHTS = Path(
    r"D:\WildfirePilot\weights\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

OUTPUT_PATH = Path("weights/dinov3_utae_pilot.pt")

EPOCHS = 2
LEARNING_RATE = 1e-4
MAX_SAMPLES = 20
POS_WEIGHT = 20.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# CHECK FILES
# =========================
if not DINO_REPO.exists():
    raise FileNotFoundError(f"Không tìm thấy DINOv3 repo: {DINO_REPO}")

if not DINO_WEIGHTS.exists():
    raise FileNotFoundError(f"Không tìm thấy checkpoint: {DINO_WEIGHTS}")


# =========================
# LOAD PILOT DATASET
# =========================
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

n_samples = min(len(dataset), MAX_SAMPLES)

if n_samples == 0:
    raise RuntimeError("Dataset không có mẫu nào.")

print("Tổng số mẫu:", len(dataset))
print("Số mẫu dùng để huấn luyện thử:", n_samples)


# =========================
# BUILD DINOv3-U-TAE
# =========================
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


# Pilot chạy trên CPU/GPU đều được.
# Đóng băng backbone và bộ chuyển 40 kênh sang RGB
# để kiểm tra pipeline huấn luyện với chi phí thấp hơn.
for param in model.dinov3_encoder.backbone.parameters():
    param.requires_grad = False

for param in model.dinov3_encoder.channel_adapter.parameters():
    param.requires_grad = False

trainable_params = [
    param for param in model.parameters()
    if param.requires_grad
]

optimizer = AdamW(trainable_params, lr=LEARNING_RATE)


# =========================
# LOSS
# =========================
def calculate_loss(logits, target):
    pos_weight = torch.tensor(
        POS_WEIGHT, dtype=logits.dtype, device=logits.device
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)

    dice_loss = (
        1.0
        - (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    ).mean()

    return bce + 0.5 * dice_loss


# Chuỗi đầu vào pilot bắt đầu từ 15/07/2021.
first_input_date = date(2021, 7, 15)


# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()

    # Không bật chế độ train cho backbone pretrained.
    model.dinov3_encoder.backbone.eval()

    total_loss = 0.0

    print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

    for idx in range(n_samples):
        x, y = dataset[idx]

        # x: [T, C, H, W]
        # y: [H, W]
        x = x.unsqueeze(0).float().to(device)
        target = (y > 0).float().unsqueeze(0).unsqueeze(1).to(device)

        input_start = first_input_date + timedelta(days=idx)
        start_doy = input_start.timetuple().tm_yday

        positions = torch.arange(
            start_doy,
            start_doy + x.shape[1],
            dtype=torch.long,
        ).unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            x,
            batch_positions=positions,
            return_att=False,
        )

        if logits.shape != target.shape:
            raise RuntimeError(
                f"Shape không khớp: output={logits.shape}, "
                f"target={target.shape}"
            )

        loss = calculate_loss(logits, target)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss không hợp lệ tại mẫu {idx}: {loss.item()}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        print(
            f"\rMẫu {idx + 1}/{n_samples} | "
            f"Loss: {loss.item():.4f}",
            end="",
            flush=True,
        )

    mean_loss = total_loss / n_samples
    print(f"\nMean train loss: {mean_loss:.6f}")


# =========================
# SAVE CHECKPOINT
# =========================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": EPOCHS,
        "samples": n_samples,
        "backbone": "DINOv3 ViT-S/16",
        "dataset": "Wildfire pilot 2021",
    },
    OUTPUT_PATH,
)

print("\n===== HOÀN TẤT =====")
print("Checkpoint:", OUTPUT_PATH.resolve())
print("DINOv3-U-TAE đã hoàn thành lượt huấn luyện thử.")
print("Lưu ý: đây chưa phải kết quả đánh giá khả năng dự báo.")
