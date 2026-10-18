import torch
from torch.utils.data import DataLoader
from src.dataloader.FireSpreadDataset import FireSpreadDataset

# Đọc pilot do mình tự thu thập
dataset = FireSpreadDataset(
    data_dir="data/raw",
    included_fire_years=[2021],
    n_leading_observations=5,
    crop_side_length=128,
    load_from_hdf5=False,
    is_train=False,
    remove_duplicate_features=False,

    # Chỉ mượn thống kê WSTS để kiểm tra luồng tensor.
    # Chưa dùng làm cấu hình chuẩn hóa chính thức
    # cho dataset riêng.
    stats_years=[2020, 2021],
)

print("Tổng số mẫu:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError("Dataset không tạo được mẫu nào.")

# Kiểm tra một mẫu
x, y = dataset[0]

print("\n===== MỘT MẪU =====")
print("Input shape:", tuple(x.shape))
print("Target shape:", tuple(y.shape))
print("Input dtype:", x.dtype)
print("Target dtype:", y.dtype)
print("Input finite:", torch.isfinite(x).all().item())
print("Target unique:", torch.unique(y).tolist())

# Kiểm tra DataLoader batch size = 2
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

batch_x, batch_y = next(iter(loader))

print("\n===== DATALOADER BATCH =====")
print("Batch X:", tuple(batch_x.shape))
print("Batch Y:", tuple(batch_y.shape))