from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.utae_paps_models.utae import UTAE


# =====================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =====================================================

DINO_REPO = Path(r"D:\WildfirePilot\dinov3")

DINO_WEIGHTS = Path(
    r"D:\WildfirePilot\weights"
    r"\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)


# =====================================================
# 2. KIỂM TRA FILE
# =====================================================

if not DINO_REPO.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy repository DINOv3: {DINO_REPO}"
    )

if not DINO_WEIGHTS.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy checkpoint: {DINO_WEIGHTS}\n"
        "Hãy kiểm tra lại tên và phần mở rộng của file."
    )


# =====================================================
# 3. CHỌN DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# 4. KHỞI TẠO DINOv3-U-TAE
# =====================================================

model = UTAE(
    input_dim=40,

    # Giữ các kích thước đặc trưng của U-TAE hiện tại
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

    # Chọn Encoder mới
    spatial_backbone="dinov3",
    dinov3_repo_dir=str(DINO_REPO),
    dinov3_weights_path=str(DINO_WEIGHTS),
).to(device)

model.train()

print("Đã khởi tạo DINOv3-U-TAE.")


# =====================================================
# 5. TẠO BATCH GIẢ ĐỂ KIỂM TRA KIẾN TRÚC
# =====================================================

# Shape giống batch dữ liệu pilot của m:
# [B, T, C, H, W]
#
# B = 1
# T = 5 ngày
# C = 40 kênh
# H = 288
# W = 192

x = torch.randn(
    1, 5, 40, 288, 192,
    dtype=torch.float32,
    device=device,
)

# Ngày trong năm của chuỗi 15–19/07/2021
batch_positions = torch.tensor(
    [[196, 197, 198, 199, 200]],
    dtype=torch.long,
    device=device,
)


# =====================================================
# 6. FORWARD: ENCODER -> L-TAE -> DECODER
# =====================================================

model.zero_grad(set_to_none=True)

logits = model(
    x,
    batch_positions=batch_positions,
    return_att=False,
)

print("\n===== FORWARD TEST =====")
print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(logits.shape))
print("Output finite:", torch.isfinite(logits).all().item())

expected_shape = (1, 1, 288, 192)

if tuple(logits.shape) != expected_shape:
    raise RuntimeError(
        f"Output shape sai. "
        f"Mong đợi {expected_shape}, "
        f"nhận được {tuple(logits.shape)}."
    )

if not torch.isfinite(logits).all():
    raise RuntimeError("Output có NaN hoặc Inf.")


# =====================================================
# 7. BACKWARD TEST
# =====================================================

# Chỉ dùng nhãn giả để kiểm tra gradient,
# không dùng kết quả này để đánh giá mô hình.
target = torch.zeros_like(logits)

loss = F.binary_cross_entropy_with_logits(
    logits,
    target,
)

loss.backward()

adapter_grad = (
    model.dinov3_encoder
    .channel_adapter.weight.grad
)

print("\n===== BACKWARD TEST =====")
print("Loss:", loss.item())
print("Adapter gradient exists:", adapter_grad is not None)

if adapter_grad is None:
    raise RuntimeError(
        "Không có gradient truyền về channel_adapter."
    )

if not torch.isfinite(adapter_grad).all():
    raise RuntimeError("Gradient có NaN hoặc Inf.")


# =====================================================
# 8. KẾT LUẬN
# =====================================================

print("\nPASS: DINOv3-U-TAE đã chạy forward và backward.")
print("Output shape:", tuple(logits.shape))