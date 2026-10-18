import ee
import yaml
import tqdm

from DataPreparation.DatasetPrepareService import DatasetPrepareService


if __name__ == "__main__":

    # Đọc cấu hình sự kiện pilot đã tạo
    with open(
        "config/pilot_fire_2021.yml",
        "r",
        encoding="utf8",
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    GEE_PROJECT_ID = "wildfire-ee-project-512003"

    ee.Initialize(project=GEE_PROJECT_ID)

    # Thêm 4 ngày trước và sau khoảng thời gian sự kiện
    N_BUFFER_DAYS = 4

    # Các khóa cấu hình không phải mã sự kiện
    non_fire_keys = [
        "output_drive_folder",
        "rectangular_size",
        "year",
    ]

    fire_names = [
        key for key in config
        if key not in non_fire_keys
    ]

    failed_locations = []

    for location in tqdm.tqdm(fire_names):
        print(f"\nĐang xử lý sự kiện: {location}")

        dataset_pre = DatasetPrepareService(
            location=location,
            config=config,
        )

        try:
            # Sự kiện pilot 25295804 thuộc UTM zone 11N
            dataset_pre.extract_dataset_from_gee_to_drive(
                "32611",
                n_buffer_days=4,
            )

        except Exception as e:
            failed_locations.append(location)
            print(f"LỖI tại {location}: {e}")

    print("\nCác sự kiện bị lỗi:", failed_locations)
    