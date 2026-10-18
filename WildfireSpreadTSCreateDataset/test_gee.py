import ee

PROJECT_ID = "wildfire-ee-project-512003"

ee.Initialize(project=PROJECT_ID)

print(
    "Earth Engine:",
    ee.String("Connected successfully").getInfo()
)

collection = ee.FeatureCollection(
    "JRC/GWIS/GlobFire/v2/FinalPerimeters"
)

print(
    "Accessible GlobFire collection:",
    collection.limit(1).size().getInfo()
)