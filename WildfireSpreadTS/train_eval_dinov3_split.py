from pathlib import Path
from datetime import date, timedelta

import torch
import torch.nn.functional as F

from src.dataloader.FireSpreadDataset import FireSpreadDataset
from src.models.utae_paps_models.utae import UTAE

torch.manual_seed(42)

ROOT = Path(__file__).resolve().parent
DINO_REPO = Path(r"D:\WildfirePilot\dinov3")
DINO_WEIGHTS = Path(
    r"D:\WildfirePilot\weights\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
OUTPUT = ROOT / "weights" / "dinov3_utae_eval.pt"

EPOCHS = 5
TRAIN_COUNT = 16
POS_WEIGHT = 20.0
FIRST_INPUT_DATE = date(2021, 7, 15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not DINO_REPO.exists():
    raise FileNotFoundError(f"Không tìm thấy: {DINO_REPO}")
if not DINO_WEIGHTS.exists():
    raise FileNotFoundError(f"Không tìm thấy: {DINO_WEIGHTS}")

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

if len(dataset) < TRAIN_COUNT + 1:
    raise RuntimeError(
        f"Cần ít nhất {TRAIN_COUNT + 1} mẫu, hiện có {len(dataset)}."
    )

train_indices = list(range(TRAIN_COUNT))
test_indices = list(range(TRAIN_COUNT, len(dataset)))

print("Tổng mẫu:", len(dataset))
print("Train:", len(train_indices))
print("Test theo thời gian:", len(test_indices))


def get_input_positions(idx, sequence_length):
    start = FIRST_INPUT_DATE + timedelta(days=idx)
    start_doy = start.timetuple().tm_yday
    return torch.arange(
        start_doy,
        start_doy + sequence_length,
        dtype=torch.long,
    ).unsqueeze(0).to(device)


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

# Giữ backbone pretrained và channel adapter cố định.
for p in model.dinov3_encoder.backbone.parameters():
    p.requires_grad = False

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)


def loss_fn(logits, target):
    pos_weight = torch.tensor(
        POS_WEIGHT, device=device, dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = 1 - (2 * intersection + 1) / (denominator + 1)

    return bce + 0.5 * dice.mean()


# Huấn luyện chỉ trên 16 mẫu đầu.
for epoch in range(EPOCHS):
    model.train()
    model.dinov3_encoder.backbone.eval()
    total_loss = 0.0

    for idx in train_indices:
        x, y = dataset[idx]
        x = x.unsqueeze(0).float().to(device)
        target = (y > 0).float().unsqueeze(0).unsqueeze(1).to(device)

        positions = get_input_positions(idx, x.shape[1])

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, batch_positions=positions, return_att=False)

        if logits.shape != target.shape:
            raise RuntimeError(
                f"Shape không khớp: output={logits.shape}, target={target.shape}"
            )

        loss = loss_fn(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss không hợp lệ ở mẫu {idx}: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Mean train loss: {total_loss / len(train_indices):.6f}"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "epochs": EPOCHS,
        "train_indices": train_indices,
        "test_indices": test_indices,
    },
    OUTPUT,
)
print("Checkpoint:", OUTPUT)


# Đánh giá trên 4 mẫu cuối chưa dùng huấn luyện.
model.eval()
tp = fp = fn = tn = 0
test_loss = 0.0

with torch.no_grad():
    for idx in test_indices:
        x, y = dataset[idx]
        x = x.unsqueeze(0).float().to(device)
        target = (y > 0).float().unsqueeze(0).unsqueeze(1).to(device)
        positions = get_input_positions(idx, x.shape[1])

        logits = model(x, batch_positions=positions, return_att=False)
        probs = torch.sigmoid(logits)
        pred = probs >= 0.5
        actual = target >= 0.5

        test_loss += F.binary_cross_entropy_with_logits(
            logits, target
        ).item()

        tp += int((pred & actual).sum().item())
        fp += int((pred & ~actual).sum().item())
        fn += int((~pred & actual).sum().item())
        tn += int((~pred & ~actual).sum().item())

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)
iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0

print("\n===== PILOT HELD-OUT EVALUATION =====")
print("Test samples:", len(test_indices))
print(f"Test BCE loss: {test_loss / len(test_indices):.6f}")
print(f"Pixel accuracy: {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-score:       {f1:.4f}")
print(f"IoU:            {iou:.4f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print("\nLưu ý: chỉ có 4 mẫu test từ một sự kiện cháy.")
print("Đây là đánh giá pilot, chưa đại diện cho khả năng tổng quát hóa.")
