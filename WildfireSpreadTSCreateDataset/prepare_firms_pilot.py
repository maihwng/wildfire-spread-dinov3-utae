from pathlib import Path
import pandas as pd

INPUT_CSV = Path("data/source/firms_viirs.csv")
OUTPUT_CSV = Path("data/pilot/viirs_active_fire_pilot.csv")

# Sự kiện pilot 25295804
LAT_MIN, LAT_MAX = 48.270141, 49.270141
LON_MIN, LON_MAX = -114.951774, -113.951774

DATE_START = "2021-07-15"
DATE_END = "2021-08-08"

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Không tìm thấy {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip().str.lower()

required = [
    "latitude", "longitude",
    "acq_date", "acq_time", "confidence", "type"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"CSV thiếu các cột: {missing}")

# Chuyển kiểu dữ liệu
for col in ["latitude", "longitude", "type"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["acq_date"] = pd.to_datetime(
    df["acq_date"], errors="coerce"
)

# Lọc theo khu vực và thời gian pilot
mask = (
    df["latitude"].between(LAT_MIN, LAT_MAX)
    & df["longitude"].between(LON_MIN, LON_MAX)
    & df["acq_date"].between(DATE_START, DATE_END)
    & df["type"].eq(0)  # presumed vegetation fire
)

df = df.loc[mask].copy()

if df.empty:
    raise RuntimeError(
        "Không còn điểm cháy nào sau khi lọc. "
        "Kiểm tra vùng, ngày và dữ liệu CSV."
    )

# Chuẩn hóa ngày
df["acq_date"] = df["acq_date"].dt.strftime("%Y-%m-%d")

# Chuẩn hóa giờ: 0959 -> 959 dưới dạng số.
# Tạo acq_hour để tương thích với logic FirePred.py.
time_text = (
    df["acq_time"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(4)
)

if not time_text.str.fullmatch(r"\d{4}").all():
    raise ValueError("Có giá trị acq_time không hợp lệ.")

df["acq_time"] = time_text.astype(int)
df["acq_hour"] = df["acq_time"]

df["confidence"] = (
    df["confidence"].astype(str).str.strip().str.lower()
)

# Chỉ giữ những thuộc tính cần thiết cho raster active-fire.
out = df[
    [
        "longitude",
        "latitude",
        "acq_date",
        "acq_time",
        "acq_hour",
        "confidence",
    ]
].drop_duplicates()

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_CSV, index=False)

print("Số điểm đầu vào:", len(pd.read_csv(INPUT_CSV)))
print("Số điểm sau lọc:", len(out))
print("Số ngày có phát hiện:", out["acq_date"].nunique())
print("\nPhân bố confidence:")
print(out["confidence"].value_counts())
print("\n5 dòng đầu:")
print(out.head().to_string(index=False))
print("\nĐã lưu:", OUTPUT_CSV)