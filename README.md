# DINOv3-UTAE: Spatio-Temporal Wildfire Spread Prediction

This repository implements the **DINOv3-UTAE** model architecture, integrating Meta's **DINOv3 ViT-S/16 Vision Foundation Model** as a spatial encoder within a **U-TAE (U-Net with Temporal Attention Encoder)** framework. Combined with a 17-channel input adapter, a multi-scale feature pyramid, and L-TAE temporal self-attention, the model is designed to predict next-day wildfire spread probability maps at $128 \times 128$ spatial resolution.

---

## 1. Project Overview

- **Core Objective**: The architecture is designed to predict active wildfire spread for the next 24 hours based on an input time-series sequence of $T$ consecutive days of multi-modal satellite, topographical, and meteorological variables.
- **Current Scope**:
  - Implementation of the **DINOv3-UTAE** hybrid model in `models/dinov3_utae.py`.
  - Spatial feature extraction using pretrained **DINOv3 ViT-S/16** (`checkpoints/vits16.pth`).
  - Handling 17 multi-modal input channels via `WildfireInputAdapter` (or weight expansion).
  - Multi-scale intermediate feature projections (Blocks 2, 5, 8, 11) with transposed convolutions to provide skip connections for the U-TAE decoder.
  - Multi-temporal aggregation via **L-TAE** and **Temporal Aggregator** with relative temporal positional encoding $[1, ..., T]$.
  - Implementation of specialized loss functions for severe class imbalance (`losses.py`: Focal Loss, Dice Loss, Combined Loss).
  - Verification of tensor shapes, forward inference, and gradient backpropagation using synthetic tensors (`test_models.py`). Full model training on real wildfire events is pending.

---

## 2. System Overview

The system processes multi-temporal satellite observations and environmental indicators to predict the spatial probability map of active wildfire spread for the subsequent day. The end-to-end pipeline operates through five coordinated stages:

1. **Input Channel Adaptation**:
   - The input is structured as a 5D spatio-temporal tensor of shape `(B, T, 17, 128, 128)`, representing a batch of $B$ sequences across $T$ consecutive observation days, with 17 physical channels at $128 \times 128$ spatial resolution.
   - For each observation day $t \in [1, \dots, T]$, the 17 raw channels (terrain, weather, vegetation, and active fire masks) are projected through a lightweight non-linear convolutional adapter (`WildfireInputAdapter`). This maps the 17-channel physical state into a 3-channel pseudo-RGB latent representation, ensuring complete compatibility with the pretrained DINOv3 vision foundation model without corrupting its learned feature distribution.

2. **Dual-Branch Spatial Feature Extraction**:
   - **Global Foundation Branch (DINOv3 ViT-S/16)**: Each adapted 3-channel frame is processed through Meta's pretrained DINOv3 ViT-S/16 backbone. Intermediate feature maps are extracted from four distinct transformer block depths (Blocks 2, 5, 8, and 11) at a spatial resolution of $8 \times 8$ (384 channels). Dedicated transposed convolution (deconvolution) layers project these features into multi-resolution representations ($64 \times 64$, $32 \times 32$, $16 \times 16$, and $8 \times 8$) to serve as multi-scale skip connections.
   - **Local Detail Branch (High-Resolution CNN Stem)**: Concurrently, a parallel 5-stage convolutional stem (`stem0`, `down1` through `down4`) extracts multi-scale local features directly from the raw 17-channel input at resolutions $128 \times 128$, $64 \times 64$, $32 \times 32$, $16 \times 16$, and $8 \times 8$. These stem features are additively fused with the projected DINOv3 representations, ensuring that fine-grained geometric fire boundaries and thin firelines are preserved alongside DINOv3's broad global semantic context.

3. **Temporal Attention Fusion (L-TAE)**:
   - At the deepest bottleneck resolution ($8 \times 8$, 384 channels), the sequence of $T$ multi-temporal feature maps is fed into the Lightweight Temporal Attention Encoder (`LTAE2d`).
   - Using 16 attention heads and a learned Master Query vector, L-TAE evaluates the temporal dynamics across the $T$ observation days. A relative temporal positional encoding $[1, \dots, T]$ informs the model of the chronological order, prioritizing recent fire activity.
   - L-TAE outputs a temporally collapsed bottleneck representation of shape `(B, 384, 8, 8)` along with a multi-head temporal attention map $\mathbf{att}$ of shape `(16, B, T, 8, 8)`.

4. **Temporal Skip Connection Aggregation**:
   - To collapse the temporal dimension of the multi-scale skip connections, the temporal attention weights $\mathbf{att}$ from L-TAE are bilinearly resampled to match each skip resolution ($16 \times 16$, $32 \times 32$, $64 \times 64$, and $128 \times 128$).
   - The `Temporal_Aggregator` module computes an attention-weighted sum over the $T$ time steps for each skip level, grouping channels across heads to produce unified 2D spatial skip maps `(B, C, H, W)` that prioritize the most informative historical dates.

5. **Multi-Stage Spatial Decoding (U-TAE Decoder)**:
   - A 4-stage convolutional decoder progressively restores spatial resolution from the fused bottleneck ($8 \times 8 \to 16 \times 16 \to 32 \times 32 \to 64 \times 64 \to 128 \times 128$).
   - At each stage, an `UpConvBlock` applies transposed convolution, concatenates the corresponding temporally aggregated skip connection, and refines the features through convolutional layers.
   - A final convolutional output block projects the 32-channel feature map at $128 \times 128$ to a single-channel logit map `(B, 1, 128, 128)`, representing the predicted probability of wildfire spread for the next day.

### Execution Contexts
- **Inference Mode**: The model accepts 5D sequences or single 4D frames and outputs both the prediction map and the temporal attention weights for explainable AI analysis and visualization.
- **Training Mode**: The model is trained using `CombinedLoss` (Focal Loss + Dice Loss) to handle severe class imbalance, with support for either end-to-end backpropagation or parameter-efficient fine-tuning with a frozen DINOv3 backbone.

---

## 3. Environment

The codebase has been verified on the following environment:

| Component | Tested Version / Specification | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Linux (Ubuntu 20.04/22.04) | Verified locally on Windows |
| **Python** | `3.11.x` / `3.13.x` | Verified with Python 3.11.9 |
| **PyTorch** | `2.11.0+cpu` (or PyTorch $\ge 2.0.0$) | Runs on CPU; CUDA enabled when GPU is available |
| **Model Checkpoint** | `checkpoints/vits16.pth` (~82.5 MB) | Checkpoint for DINOv3 ViT-S/16 (`lvd1689m`) |
| **Hardware** | Minimum 4-Core CPU, 8 GB RAM | Verified for local smoke testing and forward passes |

---

## 4. Setup

All commands and configurations use relative paths. No machine-specific or absolute directories are required.

### Step 1: Directory Structure
Organize your project directory as follows:
```text
project_root/
├── dinov3/                     # Cloned from github.com/facebookresearch/dinov3
├── utae-paps/                  # Cloned from github.com/VSainteuf/utae-paps
└── wildfire-prediction/        # Active wildfire prediction codebase
    ├── checkpoints/
    │   ├── .gitkeep
    │   └── vits16.pth          # Pretrained DINOv3 ViT-S/16 weights
    ├── models/
    │   └── dinov3_utae.py      # DINOv3-UTAE model architecture
    ├── losses.py               # Focal, Dice, and Combined loss functions
    ├── test_models.py          # Automated verification test suite
    ├── requirements.txt        # Python package dependencies
    ├── .gitignore              # Git ignore rules
    └── README.md
```

### Step 2: Install Dependencies
```bash
cd wildfire-prediction

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Checkpoint Placement
Ensure the pretrained DINOv3 ViT-S/16 checkpoint is placed at `checkpoints/vits16.pth` (the code also automatically recognizes `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`):
```bash
python -c "import os; print('Checkpoint found:', os.path.isfile('checkpoints/vits16.pth'))"
```
**Expected Output**:
```text
Checkpoint found: True
```

---

## 5. Verification & Testing

To verify that the model compiles, loads weights, performs forward passes, and handles gradient backpropagation, run the automated test suite:

```bash
python test_models.py
```

**Verification Output (Evaluated on Synthetic Tensors)**:
```text
============================================================
RUNNING DINOv3-UTAE TEST SUITE
============================================================

--- Test 1: Multi-temporal Input (B=2, T=5, C=17, H=128, W=128) ---
[DINOv3-UTAE] Loading pretrained weights from: checkpoints/vits16.pth
[DINOv3-UTAE] Checkpoint loaded successfully (<All keys matched successfully>).
Output shape: torch.Size([2, 1, 128, 128])
Attn shape:   torch.Size([16, 2, 5, 8, 8])
Test 1 PASSED.

--- Test 2: Single-day 4D Input (B=2, C=17, H=128, W=128) ---
[DINOv3-UTAE] Loading pretrained weights from: checkpoints/vits16.pth
[DINOv3-UTAE] Checkpoint loaded successfully (<All keys matched successfully>).
Output shape: torch.Size([2, 1, 128, 128])
Test 2 PASSED.

--- Test 3: Backward Pass & Gradient Flow with Loss ---
[DINOv3-UTAE] Loading pretrained weights from: checkpoints/vits16.pth
[DINOv3-UTAE] Checkpoint loaded successfully (<All keys matched successfully>).
Loss value: 0.6718
Input gradient norm: 0.0161
Test 3 PASSED.

--- Test 4: Channel Strategy 'conv1x1' ---
[DINOv3-UTAE] Loading pretrained weights from: checkpoints/vits16.pth
[DINOv3-UTAE] Checkpoint loaded successfully (<All keys matched successfully>).
Output shape with conv1x1: torch.Size([1, 1, 128, 128])
Test 4 PASSED.

--- Test 5: Frozen DINOv3 Backbone ---
[DINOv3-UTAE] Loading pretrained weights from: checkpoints/vits16.pth
[DINOv3-UTAE] Checkpoint loaded successfully (<All keys matched successfully>).
DINOv3 trainable params: 0
Total trainable params:  12,736,332
Test 5 PASSED.

============================================================
ALL TESTS COMPLETED SUCCESSFULLY!
============================================================
```

> **Note on Test Results**: The loss value `0.6718` reported in Test 3 is computed on **synthetic random tensors** (`torch.randn`) strictly to confirm non-zero gradient backpropagation and numerical stability. It does not represent actual wildfire prediction accuracy. Real training and validation on wildfire datasets is subject to subsequent training experiments.

---

## 6. Usage Examples

### Forward Inference
```python
import torch
from models.dinov3_utae import build_dinov3_utae

# Initialize model and load weights using relative path
model = build_dinov3_utae(
    in_channels=17,
    n_classes=1,
    checkpoint_path="checkpoints/vits16.pth",
    pretrained=True,
    freeze_backbone=False,
)
model.eval()

# Dummy sequence: batch=2, T=5 days, 17 channels, 128x128
dummy_input = torch.randn(2, 5, 17, 128, 128)

with torch.no_grad():
    prediction, attention_maps = model(dummy_input, return_att=True)

print("Output shape:", prediction.shape)       # (2, 1, 128, 128)
print("Attention shape:", attention_maps.shape) # (16, 2, 5, 8, 8)
```

### Training Step (Single Batch)
```python
import torch
from models.dinov3_utae import build_dinov3_utae
from losses import CombinedLoss

model = build_dinov3_utae(in_channels=17, n_classes=1)
criterion = CombinedLoss(w_focal=1.0, w_dice=1.0, alpha=0.75, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Sample synthetic batch
x = torch.randn(4, 5, 17, 128, 128)
y = torch.randint(0, 2, (4, 1, 128, 128)).float()

optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print(f"Step Loss: {loss.item():.4f}")
```

---

## 7. Model Architecture: Integrating DINOv3 into U-TAE

### 7.1. Motivation
The original **U-TAE** uses convolutional encoders (or ResNet-18) whose receptive fields grow slowly with depth. In wildfire modeling, physical drivers such as wind gusts, topographic corridors, and regional dryness operate over long spatial distances. 

We integrate Meta's **DINOv3 ViT-S/16** foundation model (21.7M parameters, pretrained on 1.689B images with Rotary Position Embeddings) to extract comprehensive global contextual representations.

### 7.2. Realistic 17-Channel Multi-Modal Input
The model accepts 17 remote sensing, weather, and topographical channels relevant to wildfire spread:
1. `active_fire`: Active fire mask at time $t$ (derived from VIIRS/MODIS detection)
2. `elevation`: Digital elevation model (SRTM/USGS)
3. `slope`: Terrain slope derived from DEM
4. `aspect_sin`: Sine component of terrain aspect
5. `aspect_cos`: Cosine component of terrain aspect
6. `ndvi`: Normalized Difference Vegetation Index (vegetation greenness / fuel load)
7. `evi`: Enhanced Vegetation Index (canopy cover / live fuel moisture)
8. `viirs_m11`: VIIRS reflectance band M11 (shortwave infrared)
9. `viirs_i4`: VIIRS brightness temperature band I4 (thermal detection)
10. `viirs_i5`: VIIRS thermal band I5
11. `total_precipitation`: Observed surface precipitation
12. `forecast_precipitation`: Short-term forecasted precipitation
13. `temperature_max`: Maximum surface temperature (2m)
14. `temperature_min`: Minimum surface temperature (2m)
15. `specific_humidity`: Specific surface humidity
16. `wind_speed`: 10m surface wind velocity
17. `wind_direction`: 10m surface wind direction

To feed these 17 channels into DINOv3 (which expects 3 channels), `models/dinov3_utae.py` implements:
- **`channel_strategy="adapter"` (Default)**: A dedicated `WildfireInputAdapter` (`Conv2d(17, 32, 3)` $\to$ `GroupNorm` $\to$ `GELU` $\to$ `Conv2d(32, 3, 1)` $\to$ `GroupNorm` $\to$ `GELU`). This projects the 17 physical channels into a 3-channel pseudo-RGB space, allowing **100% of pretrained DINOv3 weights (`vits16.pth`), including `patch_embed.proj`, to load with strict key matching**.
- **`channel_strategy="conv1x1"`**: A direct $1 \times 1$ learned linear projection.
- **`channel_strategy="expand"`**: Expands `patch_embed.proj` to 17 channels, initializing the extra 14 channels with variance-scaled weights from the 3 pretrained channels.

### 7.3. Multi-Scale Feature Pyramid & CNN Stem
On a $128 \times 128$ input with patch size 16, DINOv3 outputs an $8 \times 8$ grid of 384-dimensional tokens. To provide multi-resolution skip connections for the U-TAE decoder, `models/dinov3_utae.py` implements:
- **Intermediate Block Harvesting**: Retrieves representations from Blocks 2, 5, 8, and 11 via `dino.get_intermediate_layers(..., n=[2, 5, 8, 11], reshape=True)`.
- **Transposed Convolution Projections**:
  - Block 2 ($8 \times 8$) $\xrightarrow{\text{Deconv } 8\times}$ $64 \times 64$ (64 channels)
  - Block 5 ($8 \times 8$) $\xrightarrow{\text{Deconv } 4\times}$ $32 \times 32$ (128 channels)
  - Block 8 ($8 \times 8$) $\xrightarrow{\text{Deconv } 2\times}$ $16 \times 16$ (256 channels)
  - Block 11 ($8 \times 8$) $\xrightarrow{\text{Conv } 1\times 1}$ $8 \times 8$ (384 channels - Bottleneck)
- **High-Resolution CNN Stem**: A parallel convolutional encoder (`stem0`, `down1`, `down2`, `down3`, `down4`) extracts local spatial features ($128 \times 128 \to 64 \times 64 \to 32 \times 32 \to 16 \times 16 \to 8 \times 8$) and fuses additively with the projected DINOv3 features, preserving sharp fire perimeter boundaries.

### 7.4. Spatio-Temporal Fusion with L-TAE
- **L-TAE Attention**: At the $8 \times 8$ bottleneck (384 channels), `LTAE2d` computes multi-head attention (16 heads) using a learned Master Query vector across the $T$ input days.
- **Relative Temporal Positional Encoding**: The model uses relative day indices $\bar{t} \in [1, \dots, T]$ so that the most recent observation day receives appropriate temporal weighting.
- **Temporal Aggregator**: Attention maps $\mathbf{att}$ are bilinearly resampled to match each skip resolution ($16 \times 16, 32 \times 32, 64 \times 64, 128 \times 128$) to compute attention-weighted temporal sums of the skip features.

### 7.5. U-TAE Decoder
Four successive `UpConvBlock` stages perform transposed convolution, skip concatenation, and convolution:
$$\text{Bottleneck: } 8 \times 8 \xrightarrow{\text{Up}} 16 \times 16 \xrightarrow{\text{Up}} 32 \times 32 \xrightarrow{\text{Up}} 64 \times 64 \xrightarrow{\text{Up}} 128 \times 128$$
The final convolution outputs single-channel logits of shape $(B, 1, 128, 128)$.

---

### 7.6. Loss Functions (`losses.py`)

In wildfire remote sensing, burned pixels often account for $< 2\%$ of the spatial area. Standard Binary Cross-Entropy (BCE) struggles with this severe class imbalance. We implement:

1. **Binary Focal Loss (`FocalLoss`)**:
   $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Down-weights well-classified background pixels ($\gamma = 2.0$) and scales the positive class ($\alpha = 0.75$), focusing learning on difficult fire boundaries.

2. **Soft Dice Loss (`DiceLoss`)**:
   $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$
   Directly optimizes region overlap (Dice / F1-score) to prevent fragmented perimeter predictions.

3. **Combined Loss (`CombinedLoss`)**:
  **Combined Loss Function**

```math
\mathcal{L}_{\mathrm{Combined}} = w_{\mathrm{focal}} \cdot \mathcal{L}_{\mathrm{Focal}} + w_{\mathrm{dice}} \cdot \mathcal{L}_{\mathrm{Dice}}
```

This combines pixel-level discrimination (Focal Loss) with region-level overlap (Dice Loss).

Default weights: $w_{\mathrm{focal}} = 1.0$ and $w_{\mathrm{dice}} = 1.0$.

4. **Soft Jaccard Loss (`JaccardLoss`)**: Optimizes the Intersection-over-Union (IoU) metric directly.

---

## 8. Troubleshooting

| Symptom / Error | Probable Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: checkpoints/vits16.pth` | Checkpoint file is not in `checkpoints/` | Place `vits16.pth` (or `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`) inside `checkpoints/` |
| `ModuleNotFoundError: No module named 'dinov3'` | `dinov3` directory is not in path | Ensure `dinov3` and `wildfire-prediction` are located in the same parent workspace; `dinov3_utae.py` will resolve it automatically |
| `CUDA Out of Memory (OOM)` | Batch size is too large for GPU VRAM | Reduce batch size to 2 or 4, or enable `freeze_backbone=True` to freeze DINOv3 parameters |
| `AssertionError: Expected 17 channels, got C` | Input tensor has mismatched channel count | Verify input data channels or set `in_channels=C` when instantiating `DINOv3_UTAE` |

---

## References

- **U-TAE:** https://github.com/VSainteuf/utae-paps
- **DINOv3:** https://github.com/facebookresearch/dinov3

This project integrates a pretrained DINOv3 ViT-S/16 encoder
into a U-TAE-based spatio-temporal architecture for
next-day wildfire prediction.

Please refer to the original repositories and comply with
their respective licenses.

### Academic Citations

```bibtex
@article{garnot2021panoptic,
  title={Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks},
  author={Sainte Fare Garnot, Vivien and Landrieu, Loic},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={4872--4881},
  year={2021}
}

@article{garnot2020satellite,
  title={Satellite Image Time Series Classification with Lightweight Temporal Self-Attention},
  author={Sainte Fare Garnot, Vivien and Landrieu, Loic and Giordano, Sebastien and Chehata, Nesrine},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={615--625},
  year={2020}
}

@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104}
}
```

