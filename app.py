from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import tempfile

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dental Caries AI Analyzer",
    page_icon="🦷",
    layout="wide"
)

# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(
            32, 1, kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# LOAD MODEL
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path(__file__).parent / "dental_caries_unet_final.pth"

@st.cache_resource
def load_model():

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


model = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    original = np.array(image.convert("L"))

    # Resize to model input
    resized = image.convert("L").resize(
        (768, 768)
    )

    image_array = np.array(
        resized
    ).astype(np.float32) / 255.0

    tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output
        )

    probability = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    return original, probability


# ============================================================
# CREATE MASK + REGIONS
# ============================================================

def analyze(probability, threshold):

    mask = (
        probability >= threshold
    ).astype(np.uint8)

    total_pixels = mask.size

    positive_pixels = int(
        mask.sum()
    )

    area_percent = (
        positive_pixels /
        total_pixels
    ) * 100

    # Connected components
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    regions = []

    for i in range(1, num_labels):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        # Remove tiny noise
        if area < 20:
            continue

        x = int(
            stats[i, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[i, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[i, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[i, cv2.CC_STAT_HEIGHT]
        )

        percentage = (
            area / total_pixels
        ) * 100

        regions.append({
            "area": area,
            "percentage": percentage,
            "x": x,
            "y": y,
            "width": w,
            "height": h
        })

    regions.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    return (
        mask,
        positive_pixels,
        area_percent,
        regions
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(image, mask):

    image_rgb = np.stack(
        [image] * 3,
        axis=-1
    ).astype(np.float32)

    # Normalize image
    image_rgb = (
        image_rgb -
        image_rgb.min()
    ) / (
        image_rgb.max() -
        image_rgb.min() +
        1e-8
    )

    # Highlight caries regions
    overlay = image_rgb.copy()

    overlay[mask == 1] = [
        1.0,
        1.0,
        0.0
    ]

    # Blend
    result = (
        0.65 * image_rgb +
        0.35 * overlay
    )

    return result


# ============================================================
# REPORT
# ============================================================

def create_report(
    threshold,
    positive_pixels,
    total_pixels,
    area_percent,
    regions
):

    report = []

    report.append(
        "DENTAL CARIES AI ANALYSIS REPORT"
    )

    report.append("=" * 60)

    report.append(
        f"Threshold: {threshold:.2f}"
    )

    report.append(
        f"Total pixels: {total_pixels:,}"
    )

    report.append(
        f"Predicted caries pixels: {positive_pixels:,}"
    )

    report.append(
        f"Predicted caries area: {area_percent:.4f}%"
    )

    report.append(
        f"Detected regions: {len(regions)}"
    )

    report.append("")
    report.append(
        "INDIVIDUAL REGIONS"
    )

    report.append(
        "-" * 60
    )

    for i, region in enumerate(
        regions,
        start=1
    ):

        report.append(
            f"Region {i}: "
            f"{region['area']:,} pixels | "
            f"{region['percentage']:.4f}% | "
            f"Location: "
            f"({region['x']}, {region['y']}) | "
            f"Size: "
            f"{region['width']}x"
            f"{region['height']}"
        )

    report.append("")
    report.append(
        "MODEL INFORMATION"
    )

    report.append(
        "-" * 60
    )

    report.append(
        "Architecture: U-Net"
    )

    report.append(
        "Task: Dental caries segmentation"
    )

    report.append(
        "Decision threshold: 0.80"
    )

    report.append("")
    report.append(
        "REFERENCE MODEL PERFORMANCE"
    )

    report.append(
        "Precision: 92.57%"
    )

    report.append(
        "Recall: 80.62%"
    )

    report.append(
        "F1 / Dice: 86.18%"
    )

    report.append(
        "IoU: 75.72%"
    )

    report.append("")
    report.append(
        "NOTE: This is an AI-assisted result "
        "and should be reviewed by a qualified "
        "dental professional."
    )

    return "\n".join(report)


# ============================================================
# UI
# ============================================================

st.title(
    "🦷 Dental Caries AI Analyzer"
)

st.write(
    "AI-assisted dental caries segmentation "
    "from panoramic X-ray images using U-Net."
)

st.divider()

# Sidebar
st.sidebar.header(
    "Analysis Settings"
)

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.80,
    step=0.05
)

st.sidebar.info(
    "The model was evaluated using "
    "a threshold of 0.80."
)

uploaded_file = st.file_uploader(
    "Upload a panoramic dental X-ray",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.subheader(
        "Uploaded X-ray"
    )

    st.image(
        image,
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze X-ray",
        type="primary"
    ):

        with st.spinner(
            "Analyzing X-ray..."
        ):

            original, probability = (
                predict(image)
            )

            (
                mask,
                positive_pixels,
                area_percent,
                regions
            ) = analyze(
                probability,
                threshold
            )

        st.success(
            "Analysis completed successfully."
        )

        # Resize original to model size
        resized_original = np.array(
            image.convert("L").resize(
                (768, 768)
            )
        )

        overlay = create_overlay(
            resized_original,
            mask
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.subheader(
            "Analysis Result"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Threshold",
            f"{threshold:.2f}"
        )

        col2.metric(
            "Caries Pixels",
            f"{positive_pixels:,}"
        )

        col3.metric(
            "Caries Area",
            f"{area_percent:.4f}%"
        )

        col4.metric(
            "Detected Regions",
            len(regions)
        )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        st.subheader(
            "Segmentation Results"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.image(
                resized_original,
                caption="Original X-ray",
                use_container_width=True
            )

        with c2:

            st.image(
                mask * 255,
                caption="Predicted Caries Mask",
                use_container_width=True
            )

        with c3:

            st.image(
                overlay,
                caption="Caries Overlay",
                use_container_width=True
            )

        # ----------------------------------------------------
        # REGIONS
        # ----------------------------------------------------

        st.subheader(
            "Detected Caries Regions"
        )

        if len(regions) == 0:

            st.info(
                "No significant caries region detected "
                "at the selected threshold."
            )

        else:

            for i, region in enumerate(
                regions,
                start=1
            ):

                st.write(
                    f"**Region {i}** — "
                    f"{region['area']:,} pixels | "
                    f"{region['percentage']:.4f}% | "
                    f"Location: "
                    f"({region['x']}, "
                    f"{region['y']})"
                )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        report = create_report(
            threshold,
            positive_pixels,
            mask.size,
            area_percent,
            regions
        )

        st.subheader(
            "Dental Caries Report"
        )

        st.text_area(
            "Generated Report",
            report,
            height=400
        )

        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="dental_caries_report.txt",
            mime="text/plain"
        )

        # ----------------------------------------------------
        # MODEL PERFORMANCE
        # ----------------------------------------------------

        st.subheader(
            "Model Performance"
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Precision",
            "92.57%"
        )

        p2.metric(
            "Recall",
            "80.62%"
        )

        p3.metric(
            "F1 / Dice",
            "86.18%"
        )

        p4.metric(
            "IoU",
            "75.72%"
        )

        st.caption(
            "These are the model's held-out test "
            "performance metrics, not the confidence "
            "of this individual prediction."
        )

else:

    st.info(
        "Upload a PNG, JPG, or JPEG panoramic "
        "dental X-ray to begin."
    )

st.divider()

st.caption(
    "Dental Caries AI Analyzer | "
    "U-Net based segmentation | "
    "AI-assisted analysis — not a medical diagnosis"
)
