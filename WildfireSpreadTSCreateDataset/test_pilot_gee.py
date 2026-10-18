import ee

from DataPreparation.satellites.FirePred import FirePred

PROJECT_ID = "wildfire-ee-project-512003"
ASSET_ID = (
    "projects/wildfire-ee-project-512003/"
    "assets/pilot_viirs_active_fire"
)

# Khởi tạo Google Earth Engine
ee.Initialize(project=PROJECT_ID)

# 1. Kiểm tra quyền truy cập Asset
fires = ee.FeatureCollection(ASSET_ID)

print("Số điểm trong Asset:", fires.size().getInfo())

sample = ee.Feature(fires.first())
print("Thuộc tính điểm đầu tiên:")
print(sample.toDictionary().getInfo())

# 2. Kiểm tra số điểm từng ngày sau khi loại confidence thấp
for day in ["2021-07-22", "2021-07-23"]:
    daily_fires = (
        fires
        .filter(ee.Filter.eq("acq_date", day))
        .filter(ee.Filter.neq("confidence", "l"))
    )

    print(
        f"Active-fire ngày {day}:",
        daily_fires.size().getInfo()
    )

# 3. Thử tạo raster đa nguồn cho một ngày
geometry = ee.Geometry.Rectangle([
    -114.951774,
    48.270141,
    -113.951774,
    49.270141,
])

fire_model = FirePred()

daily_images = fire_model.compute_daily_features(
    "2021-07-22T00:00",
    "2021-07-22T23:59",
    geometry,
)

print("Số ảnh đầu ra:", daily_images.size().getInfo())

image = ee.Image(daily_images.first())

print("Các band đầu ra:")
print(image.bandNames().getInfo())