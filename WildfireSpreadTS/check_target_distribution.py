from pathlib import Path
from datetime import date, timedelta

import torch
from src.dataloader.FireSpreadDataset import FireSpreadDataset

root = Path(r"data\raw\2021\fire_25295804")
files = sorted(root.glob("*.tif"))

if len(files) != 25:
    raise RuntimeError(
        f"Mong đợi 25 raster, tìm thấy {len(files)}."
    )

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
print("\n===== THỐNG KÊ NHÃN =====")

samples_with_fire = 0
total_fire_pixels = 0

for i in range(len(dataset)):
    x, y = dataset[i]

    # 5 ngày đầu vào, ngày mục tiêu là file thứ i + 5
    target_file = files[i + 5]
    target_date = target_file.name.split("_")[0]

    fire_pixels = int((y > 0).sum().item())
    unique_labels = torch.unique(y).tolist()

    if fire_pixels > 0:
        samples_with_fire += 1

    total_fire_pixels += fire_pixels

    print(
        f"Sample {i:02d} | Target: {target_date} "
        f"| Fire pixels: {fire_pixels} "
        f"| Labels: {unique_labels}"
    )

print("\n===== TỔNG KẾT =====")
print("Mẫu có pixel cháy:", samples_with_fire, "/", len(dataset))
print("Tổng pixel cháy qua các mẫu:", total_fire_pixels)

if samples_with_fire == 0:
    print(
        "CẢNH BÁO: Tất cả nhãn đều rỗng. "
        "Cần kiểm tra band active-fire và quy trình tạo nhãn."
    )
elif samples_with_fire < len(dataset):
    print(
        "Có cả mẫu có cháy và không cháy. "
        "Tiếp tục kiểm tra tỷ lệ mất cân bằng nhãn."
    )
else:
    print("Mọi mẫu đều có ít nhất một pixel cháy.")