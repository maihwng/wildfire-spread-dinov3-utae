from pathlib import Path
from datetime import date, timedelta
import rasterio

root = Path(r"data\raw\2021\fire_25295804")
files = sorted(root.glob("*.tif"))

print("Số raster:", len(files))

# Các ngày dự kiến: 15/07 đến 08/08/2021
expected_dates = {
    (date(2021, 7, 15) + timedelta(days=i)).isoformat()
    for i in range(25)
}

date_files = {}
for path in files:
    try:
        day = path.name.split("_")[0]
        date.fromisoformat(day)
        date_files.setdefault(day, []).append(path.name)
    except ValueError:
        print("Tên file không đúng định dạng ngày:", path.name)

actual_dates = set(date_files)
missing_dates = sorted(expected_dates - actual_dates)
duplicate_dates = {
    day: names
    for day, names in date_files.items()
    if len(names) > 1
}

print("Ngày bị thiếu:", missing_dates)
print("Ngày bị trùng:", duplicate_dates)

errors = []
reference = None

for path in files:
    with rasterio.open(path) as ds:
        metadata = (
            ds.width,
            ds.height,
            ds.count,
            ds.crs,
            ds.transform,
            ds.dtypes,
        )

        if reference is None:
            reference = metadata
        elif metadata != reference:
            errors.append(
                f"{path.name}: metadata khác raster đầu tiên"
            )

        if ds.count != 23:
            errors.append(
                f"{path.name}: có {ds.count} band thay vì 23"
            )

        if ds.crs != rasterio.crs.CRS.from_epsg(32611):
            errors.append(
                f"{path.name}: CRS không phải EPSG:32611"
            )

        if set(ds.dtypes) != {"float32"}:
            errors.append(
                f"{path.name}: kiểu dữ liệu không phải float32"
            )

print("\nLỗi kiểm tra:")
print("\n".join(errors) if errors else "Không có lỗi metadata.")

if reference:
    print("\nKích thước raster:", reference[0:2])
    print("Số band:", reference[2])
    print("CRS:", reference[3])
    print("Kiểu dữ liệu:", set(reference[5]))

if (
    len(files) == 25
    and not missing_dates
    and not duplicate_dates
    and not errors
):
    print("\nPASS: Đủ 25 raster, ngày liên tục, metadata nhất quán.")
else:
    print("\nChưa đạt: kiểm tra lại danh sách file và lỗi phía trên.")