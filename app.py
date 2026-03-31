# =============================================================================
# app.py — Silicon Wafer Defect Detection (9 Classes)
# No OpenCV dependency — uses PIL + NumPy only
# Run with: streamlit run app.py
# Requires: best_hybrid.pth / best_cnn.pth / best_vit.pth
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cpu")
IMG_SIZE = 64    # CNN / Hybrid input size
VIT_SIZE = 224   # ViT input size

# ── CLASS NAMES — MUST match LabelEncoder alphabetical order used in training ──
# LabelEncoder.fit_transform() on:
#   ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none']
# produces alphabetical order:
#   0=Center  1=Donut  2=Edge-Loc  3=Edge-Ring  4=Loc
#   5=Near-full  6=Random  7=Scratch  8=none
#
# ⚠️  class_names.npy saved in Cell 63 of the notebook has a WRONG order
#     (Random/Scratch/Near-full are swapped). We IGNORE that file and use
#     the correct alphabetical order below.
CLASS_NAMES = [
    "Center",     # 0
    "Donut",      # 1
    "Edge-Loc",   # 2
    "Edge-Ring",  # 3
    "Loc",        # 4
    "Near-full",  # 5
    "Random",     # 6
    "Scratch",    # 7
    "none",       # 8
]
NUM_CLASSES = len(CLASS_NAMES)

DEFECT_INFO = {
    "Center":    "Defects concentrated near the wafer center.",
    "Donut":     "Ring-shaped defect pattern around the wafer center.",
    "Edge-Loc":  "Localised defects along the wafer edge.",
    "Edge-Ring": "Full or partial ring of defects at the wafer edge.",
    "Loc":       "Localised cluster of defects in one region.",
    "Near-full": "Nearly the entire wafer surface is defective.",
    "Random":    "Randomly distributed defects with no spatial pattern.",
    "Scratch":   "Linear scratch-like defect across the wafer.",
    "none":      "No defect detected — wafer is clean.",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS — must match training architecture exactly
# ─────────────────────────────────────────────────────────────────────────────

class WaferCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, img_size=64, d_model=128,
                 nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.BatchNorm2d(32),     nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),     nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        feat_h   = img_size // 4
        seq_len  = feat_h * feat_h
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat       = self.cnn_backbone(x)
        B, C, H, W = feat.shape
        feat       = feat.flatten(2).transpose(1, 2)
        feat       = feat + self.pos_embed[:, :feat.size(1), :]
        feat       = self.transformer(feat)
        return self.head(feat.mean(dim=1))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class ViTAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = self.dropout((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B, N, C))

class ViTBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = ViTAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class WaferViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3,
                 embed_dim=192, depth=12, num_heads=3,
                 mlp_ratio=4.0, num_classes=9, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches      = self.patch_embed.num_patches
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)
        self.blocks      = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = self.pos_drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        x   = self.norm(self.blocks(x))
        return self.head(x[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "Hybrid CNN-Transformer": {
        "cls":      HybridCNNTransformer,
        "kwargs":   {"num_classes": NUM_CLASSES},
        "weights":  "best_hybrid.pth",
        "img_size": IMG_SIZE,
    },
    "CNN Baseline": {
        "cls":      WaferCNN,
        "kwargs":   {"num_classes": NUM_CLASSES},
        "weights":  "best_cnn.pth",
        "img_size": IMG_SIZE,
    },
    "Vision Transformer (ViT)": {
        "cls":      WaferViT,
        "kwargs":   {"num_classes": NUM_CLASSES},
        "weights":  "best_vit.pth",
        "img_size": VIT_SIZE,
    },
}

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    cfg   = MODEL_CONFIG[model_name]
    model = cfg["cls"](**cfg["kwargs"]).to(DEVICE)
    w     = cfg["weights"]
    if not os.path.exists(w):
        return None, f"Weight file `{w}` not found. Place it in the same folder as app.py."
    model.load_state_dict(torch.load(w, map_location=DEVICE))
    model.eval()
    return model, None


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — matches notebook's wafer_to_img() exactly, without cv2
#
# Notebook pipeline:
#   arr = np.array(wafer, dtype=np.uint8)         # values: 0, 1, 2
#   arr = (arr * 127).clip(0, 255).astype(uint8)  # → 0, 127, 254
#   img = cv2.resize(arr, (64,64), INTER_NEAREST)  # single-channel uint8
#   img = np.stack([img]*3, axis=-1)               # (64, 64, 3) uint8
#   saved as: img.astype(float32) / 255.0          # normalised to [0, ~0.996]
#   loaded as: torch.tensor(img).permute(2,0,1)    # (3, 64, 64)
#
# App replication: uploaded image → grayscale → NEAREST resize → /255 → stack 3ch
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img_pil: Image.Image, img_size: int) -> torch.Tensor:
    gray = img_pil.convert("L")
    gray = gray.resize((img_size, img_size), Image.NEAREST)
    arr  = np.array(gray, dtype=np.float32) / 255.0
    img_3ch = np.stack([arr, arr, arr], axis=0)       # (3, H, W)
    return torch.tensor(img_3ch).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, tensor: torch.Tensor):
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx    = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Silicon Wafer Defect Detection",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Silicon Wafer Defect Detection")
st.markdown(
    "Upload a **wafer map image** and choose a model to classify the defect type "
    "from **9 classes** (WM811K dataset).  \n"
    "Models: Hybrid CNN-Transformer · CNN Baseline · Vision Transformer (ViT)"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_name     = st.selectbox("Select Model", list(MODEL_CONFIG.keys()), index=0)
    conf_threshold = st.slider("Confidence highlight threshold", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    st.subheader("📋 9 Defect Classes")
    for cls, desc in DEFECT_INFO.items():
        st.markdown(f"**{cls}** — {desc}")
    st.divider()
    st.caption(
        "**Class index order** (LabelEncoder alphabetical):  \n"
        + "  \n".join(f"{i}: {c}" for i, c in enumerate(CLASS_NAMES))
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a wafer map image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Wafer Map")
        st.image(img_pil, use_column_width=True)
        st.caption(f"Original size: {img_pil.width} × {img_pil.height} px")

    with col_result:
        st.subheader(f"Prediction — {model_name}")

        with st.spinner(f"Loading {model_name}..."):
            model, err = load_model(model_name)

        if err:
            st.error(err)
            st.stop()

        cfg    = MODEL_CONFIG[model_name]
        tensor = preprocess(img_pil, cfg["img_size"])

        with st.spinner("Running inference..."):
            pred_cls, confidence, probs = predict(model, tensor)

        highlight   = confidence >= conf_threshold
        badge_color = "🟢" if pred_cls == "none" else ("🔴" if highlight else "🟡")

        st.metric("Predicted Defect", f"{badge_color} {pred_cls}")
        st.metric("Confidence", f"{confidence:.2%}")

        info = DEFECT_INFO.get(pred_cls, "")
        if info:
            st.info(f"ℹ️ {info}")

        st.divider()
        st.subheader("Class Probabilities")
        sorted_pairs = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: -x[1])
        for cls, p in sorted_pairs:
            st.write(f"**{cls}**")
            st.progress(float(p), text=f"{p:.2%}")

    st.divider()

    with st.expander("🔍 What the model sees (preprocessed input)"):
        proc_size = cfg["img_size"]
        preview   = img_pil.convert("L").resize((proc_size, proc_size), Image.NEAREST)
        st.image(preview, caption=f"Grayscale → {proc_size}×{proc_size} px (NEAREST resize)", width=200)
        st.caption(
            "All 3 input channels are identical copies of this grayscale image, "
            "normalised to [0, 1]. Matches `wafer_to_img()` in the training notebook."
        )

else:
    st.info(
        "👆 Upload a wafer map image using the file uploader above.\n\n"
        "Wafer maps from the WM811K dataset work best "
        "(pixel values: 0 = absent die, 1 = good die, 2 = defective die)."
    )
    st.subheader("🏗️ Model Architecture Summary")
    import pandas as pd
    arch_data = {
        "Model":         ["CNN Baseline",   "Hybrid CNN-Transformer",   "Vision Transformer"],
        "Weight file":   ["best_cnn.pth",   "best_hybrid.pth",          "best_vit.pth"],
        "Input size":    [f"{IMG_SIZE}×{IMG_SIZE}", f"{IMG_SIZE}×{IMG_SIZE}", f"{VIT_SIZE}×{VIT_SIZE}"],
        "Approx params": ["~1.5 M",         "~0.7 M",                   "~5.7 M"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Project by B. Reshmitha (23B81A67G5) & K. Ushaswi (23B81A67J6) — "
    "CVR College of Engineering, Dept. of CSE (Data Science)  |  "
    "Dataset: WM811K Wafer Map (Kaggle)"
)
