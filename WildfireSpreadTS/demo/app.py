from pathlib import Path
from datetime import date, timedelta
import sys

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import rasterio

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "raw" / "2021" / "fire_25295804"
CHECKPOINT = ROOT / "weights" / "dinov3_utae_pilot.pt"

DINO_REPO = REPO_ROOT / "dinov3"
DINO_WEIGHTS = (
    REPO_ROOT
    / "weights"
    / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

FIRST_INPUT_DATE = date(2021, 7, 19)
MIN_DATE = date(2021, 7, 19)
MAX_DATE = date(2021, 8, 7)

st.set_page_config(
    page_title="Dự báo cháy rừng",
    page_icon="🔥",
    layout="wide",
)

# Giao diện bám sát HTML thiết kế gốc.
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stMain"] {
    background: #111827 !important;
    color: #f3f4f6 !important;
}
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1152px;
    padding: 2rem;
}
[data-testid="stHeader"] {
    background: transparent;
}
h1, h2, h3, p, label, span {
    color: #f3f4f6;
}
h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #1e293b;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 1rem;
}
[data-testid="stSelectbox"] label,
[data-testid="stDateInput"] label {
    color: #e5e7eb !important;
    font-size: 0.9rem !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stDateInput"] input {
    background: #20324c !important;
    color: #e5e7eb !important;
    border-color: #2b4468 !important;
}
.stButton > button {
    background: #0d6efd !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    min-height: 44px;
}
.stButton > button:hover {
    background: #2563eb !important;
}
[data-testid="stImage"] img {
    border: 1px solid #374151;
    border-radius: 6px;
}
[data-testid="stAlert"] {
    border-radius: 6px;
}
small, .caption {
    color: #9ca3af !important;
}
#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:24px">
  <h1 style="margin-bottom:4px">🔥 Dự báo cháy rừng</h1>
  <p style="color:#9ca3af;font-size:14px;margin-left:36px">
    DINOv3 · U-TAE · Dự đoán cháy rừng ngày tiếp theo
  </p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div style="font-weight:600;margin-bottom:12px">'
            'Điều khiển dự báo</div>',
            unsafe_allow_html=True,
        )

        fire_event = st.selectbox(
            "Chọn vụ cháy",
            ["Vụ cháy 001"],
        )

        current_date = st.date_input(
            "Chọn ngày hiện tại (t)",
            value=date(2021, 7, 21),
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            format="DD/MM/YYYY",
        )

        next_date = current_date + timedelta(days=1)

        st.markdown(
            f'<p style="font-size:11px;color:#9ca3af;'
            f'margin-top:4px;margin-bottom:30px">'
            f'Ngày dự đoán tiếp theo: {next_date:%d/%m/%Y}'
            f'</p>',
            unsafe_allow_html=True,
        )

        predict_clicked = st.button(
            "▶  Dự đoán ngày tiếp theo",
            use_container_width=True,
        )

        st.markdown(
            '<p style="font-size:11px;color:#9ca3af;margin-top:14px">'
            'Sự kiện pilot: 25295804 · Năm 2021'
            '</p>',
            unsafe_allow_html=True,
        )


def read_active_fire(path):
    with rasterio.open(path) as src:
        if src.count < 23:
            raise ValueError(
                f"Raster chỉ có {src.count} band, cần ít nhất 23 band."
            )
        image = src.read(23).astype(np.float32)
        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "band": "active fire",
        }
    image[~np.isfinite(image)] = np.nan
    return image, metadata


def run_prediction(selected_date):
    """Chạy inference thật nếu đủ code, checkpoint và dữ liệu local."""
    import torch
    from src.dataloader.FireSpreadDataset import FireSpreadDataset
    from src.models.utae_paps_models.utae import UTAE

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {CHECKPOINT}"
        )
    if not DINO_REPO.exists():
        raise FileNotFoundError(
            f"Không tìm thấy DINOv3 repo: {DINO_REPO}"
        )
    if not DINO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Không tìm thấy trọng số DINOv3: {DINO_WEIGHTS}"
        )

    dataset = FireSpreadDataset(
        data_dir=str(ROOT / "data" / "raw"),
        included_fire_years=[2021],
        n_leading_observations=5,
        crop_side_length=128,
        load_from_hdf5=False,
        is_train=False,
        remove_duplicate_features=False,
        stats_years=[2020, 2021],
    )

    sample_idx = (selected_date - FIRST_INPUT_DATE).days
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise ValueError(
            "Ngày chọn nằm ngoài khoảng dữ liệu có thể dự đoán."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UTAE(
        input_dim=40,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 1],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
        spatial_backbone="dinov3",
        dinov3_repo_dir=str(DINO_REPO),
        dinov3_weights_path=str(DINO_WEIGHTS),
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x, _ = dataset[sample_idx]
    x = x.unsqueeze(0).float().to(device)

    input_start = selected_date - timedelta(days=4)
    start_doy = input_start.timetuple().tm_yday

    positions = torch.arange(
        start_doy,
        start_doy + x.shape[1],
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            x,
            batch_positions=positions,
            return_att=False,
        )
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return probability


# Chỉ giữ kết quả dự đoán ứng với ngày đã chọn.
if "prediction_date" not in st.session_state:
    st.session_state.prediction_date = None
if "prediction_map" not in st.session_state:
    st.session_state.prediction_map = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

if predict_clicked:
    st.session_state.prediction_date = None
    st.session_state.prediction_map = None
    st.session_state.prediction_error = None

    try:
        with st.spinner("Đang chạy DINOv3-U-TAE..."):
            result = run_prediction(current_date)
        st.session_state.prediction_date = current_date
        st.session_state.prediction_map = result
    except Exception as exc:
        st.session_state.prediction_error = str(exc)

with right:
    st.markdown(
        '<h3 style="font-size:18px;font-weight:600;margin:4px 0 12px">'
        'Bản đồ cháy rừng</h3>',
        unsafe_allow_html=True,
    )

    result_map = (
        st.session_state.prediction_map
        if st.session_state.prediction_date == current_date
        else None
    )

    if result_map is not None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.imshow(result_map, cmap="inferno", vmin=0, vmax=1)
        ax.set_axis_off()
        ax.set_title(
            f"Xác suất cháy dự đoán — {next_date:%d/%m/%Y}",
            color="white",
        )
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.warning(
            "Đây là kết quả từ checkpoint pilot đã huấn luyện 2 epoch. "
            "Chưa được đánh giá đầy đủ nên không dùng làm cảnh báo cháy "
            "thực tế."
        )

    else:
        raster_path = DATA_DIR / (
            f"{current_date.isoformat()}_fire_25295804.tif"
        )

        if raster_path.exists():
            try:
                image, metadata = read_active_fire(raster_path)
                fig, ax = plt.subplots(figsize=(8, 5.5))
                ax.imshow(image, cmap="inferno")
                ax.set_axis_off()
                ax.set_title(
                    f"Dữ liệu quan sát — {current_date:%d/%m/%Y}",
                    color="white",
                )
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.caption(
                    "Đang hiển thị band active fire của GeoTIFF thực tế. "
                    "Đây là dữ liệu quan sát, chưa phải dự báo."
                )
            except Exception as exc:
                st.error(f"Không đọc được raster: {exc}")
        else:
            st.image(
                "https://images.unsplash.com/photo-1590502593747-42a996133562"
                "?auto=format&fit=crop&w=1200&q=80",
                use_container_width=True,
            )
            st.caption(
                "Ảnh minh họa giao diện. Hãy chạy local với dữ liệu pilot "
                "để xem raster thực tế."
            )

    if st.session_state.prediction_error:
        st.warning(
            "Chưa chạy được inference: "
            + st.session_state.prediction_error
        )

st.markdown(
    '<p style="color:#6b7280;font-size:11px;margin-top:24px">'
    'WildfireSpreadTS · Pilot 2021 · DINOv3-U-TAE'
    '</p>',
    unsafe_allow_html=True,
)

