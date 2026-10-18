from pathlib import Path
import pandas as pd

# Sửa đường dẫn nếu tên CSV thực tế của m khác
input_csv = Path(
    r"data\candidate\fire_events.csv"
)

output_csv = Path(r"data\pilot\pilot_fire.csv")

if not input_csv.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file CSV: {input_csv}"
    )

df = pd.read_csv(input_csv)

required = [
    "Id", "lat", "lon", "start_date", "end_date", "area"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"CSV thiếu cột: {missing}")

# Chuyển ngày tháng
df["start_dt"] = pd.to_datetime(
    df["start_date"], errors="coerce", utc=True
)
df["end_dt"] = pd.to_datetime(
    df["end_date"], errors="coerce", utc=True
)

df["duration_days"] = (
    df["end_dt"].dt.normalize()
    - df["start_dt"].dt.normalize()
).dt.days + 1

# Chuyển tọa độ và diện tích sang kiểu số
for col in ["lat", "lon", "area"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Không giới hạn kinh độ vào riêng UTM 10N nữa
candidates = df[
    df["lat"].between(-80, 84)
    & df["lon"].between(-180, 180)
    & (df["duration_days"] >= 7)
    & (df["end_dt"] >= df["start_dt"])
    & df["Id"].notna()
    & df["area"].notna()
].copy()

if candidates.empty:
    print("Không có sự kiện kéo dài ít nhất 7 ngày.")
    raise SystemExit(1)

# Tính múi UTM dựa trên tọa độ mỗi sự kiện
candidates["utm_zone"] = (
    ((candidates["lon"] + 180) // 6 + 1)
    .clip(1, 60)
    .astype(int)
)

candidates["epsg"] = (
    32600 + candidates["utm_zone"]
).where(
    candidates["lat"] >= 0,
    32700 + candidates["utm_zone"]
)

# Ưu tiên sự kiện kéo dài lâu, sau đó diện tích lớn
candidates = candidates.sort_values(
    ["duration_days", "area"],
    ascending=[False, False]
)

print("10 sự kiện phù hợp:")
print(
    candidates[
        [
            "Id", "lat", "lon", "start_date",
            "end_date", "duration_days", "area", "epsg"
        ]
    ].head(10).to_string(index=False)
)

# Lấy một sự kiện làm pilot
selected = candidates.iloc[[0]][
    [
        "Id", "lat", "lon", "start_date",
        "end_date", "area", "duration_days",
        "utm_zone", "epsg"
    ]
].copy()

output_csv.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(output_csv, index=False)

print("\nĐã chọn sự kiện pilot:")
print(selected.to_string(index=False))
print("\nĐã lưu:", output_csv)