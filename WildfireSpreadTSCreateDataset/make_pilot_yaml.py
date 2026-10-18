from pathlib import Path
import pandas as pd

input_csv = Path("data/pilot/pilot_fire.csv")
output_yaml = Path("config/pilot_fire_2021.yml")

if not input_csv.exists():
    raise FileNotFoundError(f"Không tìm thấy: {input_csv}")

df = pd.read_csv(input_csv)

if len(df) != 1:
    raise ValueError(
        f"Pilot cần đúng 1 sự kiện, hiện có {len(df)}."
    )

row = df.iloc[0]

fire_id = f"fire_{int(row['Id'])}"
start_date = pd.to_datetime(row["start_date"]).strftime("%Y-%m-%d")
end_date = pd.to_datetime(row["end_date"]).strftime("%Y-%m-%d")

# Cấu hình hướng xuất sang Google Drive.
yaml_content = f"""output_drive_folder: WildfirePilot_2021
rectangular_size: 0.5
year: 2021

{fire_id}:
  latitude: {float(row['lat'])}
  longitude: {float(row['lon'])}
  start: {start_date}
  end: {end_date}
"""

output_yaml.parent.mkdir(parents=True, exist_ok=True)
output_yaml.write_text(yaml_content, encoding="utf-8")

print("Đã tạo:", output_yaml)
print(yaml_content)