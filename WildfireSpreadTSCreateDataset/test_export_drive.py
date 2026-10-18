import ee
from DataPreparation.satellites.FirePred import FirePred

PROJECT_ID = "wildfire-ee-project-512003"

# Khởi tạo Earth Engine
ee.Initialize(project=PROJECT_ID)

# Vùng quanh sự kiện pilot 25295804
geometry = ee.Geometry.Rectangle(
    [
        -114.951774,  # West
        48.270141,    # South
        -113.951774,  # East
        49.270141,    # North
    ],
    proj="EPSG:4326",
    geodesic=False,
)

# Tạo raster đa nguồn trong một ngày
fire_model = FirePred()

images = fire_model.compute_daily_features(
    "2021-07-22T00:00",
    "2021-07-22T23:59",
    geometry,
)

count = images.size().getInfo()
print("Số ảnh:", count)

if count != 1:
    raise RuntimeError(
        f"Mong đợi 1 ảnh, nhận được {count}."
    )

image = ee.Image(images.first()).toFloat()

bands = image.bandNames().getInfo()

print("Số band:", len(bands))
print("Các band:", bands)
print("Kiểu dữ liệu band:", image.bandTypes().getInfo())

if len(bands) != 23:
    raise RuntimeError(
        f"Mong đợi 23 band, nhận được {len(bands)}."
    )

# Gửi tác vụ xuất trực tiếp lên Google Drive
task = ee.batch.Export.image.toDrive(
    image=image,
    description="pilot_25295804_20210722",
    folder="WildfirePilot_2021_Test",
    fileNamePrefix="pilot_25295804_20210722",
    region=geometry,
    scale=375,
    crs="EPSG:32611",
    maxPixels=1e13,
    fileFormat="GeoTIFF",
)

task.start()

print("Đã gửi tác vụ xuất.")
print("Task ID:", task.id)
print("Trạng thái:", task.status())