import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV3Encoder(nn.Module):
    """
    DINOv3 ViT-S/16 làm spatial encoder cho U-TAE.

    Input:
        [B, T, C, H, W]

    Output:
        List gồm 4 feature maps:
        [B, T, 64,  H,   W]
        [B, T, 64,  H/2, W/2]
        [B, T, 64,  H/4, W/4]
        [B, T, 128, H/8, W/8]
    """

    def __init__(
        self,
        input_dim,
        encoder_widths,
        repo_dir,
        weights_path=None,
    ):
        super().__init__()

        if len(encoder_widths) != 4:
            raise ValueError(
                "encoder_widths phải có đúng 4 phần tử."
            )

        self.encoder_widths = list(encoder_widths)

        # Adapter: số kênh dataset -> 3 kênh đầu vào DINOv3.
        self.channel_adapter = nn.Conv2d(
            input_dim, 3, kernel_size=1
        )

        # Nạp checkpoint pretrained đã tải về.
        if weights_path is None:
            self.backbone = torch.hub.load(
                str(repo_dir),
                "dinov3_vits16",
                source="local",
                pretrained=False,
            )
        else:
            self.backbone = torch.hub.load(
                str(repo_dir),
                "dinov3_vits16",
                source="local",
                weights=str(weights_path),
            )

        feature_dim = 384

        # DINOv3 ViT-S/16 có embedding dimension 384.
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    feature_dim,
                    channels,
                    kernel_size=1,
                ),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            )
            for channels in self.encoder_widths
        ])

        # Các mức đặc trưng ở 4 Transformer blocks.
        self.intermediate_layers = [2, 5, 8, 11]

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                f"Input phải có dạng [B,T,C,H,W], "
                f"nhận được {tuple(x.shape)}"
            )

        batch_size, timesteps, channels, height, width = x.shape

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "H và W phải chia hết cho patch size 16."
            )

        # Gộp batch và thời gian để chạy từng ảnh qua backbone.
        x = x.reshape(
            batch_size * timesteps,
            channels,
            height,
            width,
        )

        rgb = torch.sigmoid(self.channel_adapter(x))

        mean = rgb.new_tensor(
            [0.430, 0.411, 0.296]
        ).view(1, 3, 1, 1)

        std = rgb.new_tensor(
            [0.213, 0.156, 0.143]
        ).view(1, 3, 1, 1)

        rgb = (rgb - mean) / std

        # Lấy đặc trưng ở 4 tầng của DINOv3.
        intermediate = self.backbone.get_intermediate_layers(
            rgb,
            n=self.intermediate_layers,
            reshape=True,
            norm=True,
        )

        # Giữ đúng kích thước mà Encoder U-TAE gốc cung cấp.
        target_sizes = [
            (height, width),
            (height // 2, width // 2),
            (height // 4, width // 4),
            (height // 8, width // 8),
        ]

        feature_maps = []

        for features, projection, size in zip(
            intermediate,
            self.projections,
            target_sizes,
        ):
            features = projection(features)

            features = F.interpolate(
                features,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

            features = features.reshape(
                batch_size,
                timesteps,
                features.shape[1],
                size[0],
                size[1],
            )

            feature_maps.append(features)

        return feature_maps