"""
Veritas AI: Chest X-ray Consensus Cascade
Dual-Engine: High-Performance FastAPI REST API + Interactive Gradio UI
Optimized for 100% Free Hugging Face Spaces (Gradio SDK, 16 GB RAM).
"""

import os
os.environ["GRADIO_SSR_MODE"] = "False"  # must be set before gradio is imported
import io
import json
import time
import base64
import hashlib
import logging
import asyncio
import threading
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Enforce CPU thread cap to prevent cloud container contention
import torch
torch.set_num_threads(2)

import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models

import gradio as gr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("veritas_app")

# ---------------- Constants ----------------
IMG = 224
GEM_SIDE = 768
MAX_BYTES = 15 * 1024 * 1024
DEVICE = torch.device("cpu")

CLASSES = ["NORMAL", "PNEUMONIA", "COVID-19"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

NAMES = {
    1: "Part I: Baseline",
    2: "Part II: Shortcut-Robust",
    3: "Part III: + COVID-19",
    4: "Part IV: Gemini-Filtered",
    5: "Part V: Consensus Cascade"
}

DESCRIPTION_5 = "Part V: ResNet-18 + Gemini second reader, scope and quality gates, calibrated thresholds, decision tiers A to X."

PREPROCESS_DESCRIPTIONS = {
    1: "ImageNet normalization only (no standardization)",
    2: "Per-image standardization + ImageNet normalization",
    3: "Per-image standardization + ImageNet normalization",
    4: "Per-image z-score (3-channel repeated)",
    5: "Per-image standardization + ImageNet normalization"
}

STAGE_DEFINITIONS = [
    ("upload", "Upload & Resize (224x224 Gray)"),
    ("scope_check", "Scope & Quality Audit (Gemini)"),
    ("preprocess", "Standardization"),
    ("resnet", "ResNet-18 Backbone"),
    ("calibration", "Temperature Calibration"),
    ("gradcam", "Grad-CAM Attribution"),
    ("gemini_reader", "Gemini Second Reader"),
    ("decision", "Consensus Decision Rule")
]

ACTIVE_STAGES = {p: {"upload", "preprocess", "resnet", "gradcam", "decision"} for p in range(1, 5)}
ACTIVE_STAGES[5] = {s for s, _ in STAGE_DEFINITIONS}

HF_REPOS = {part: f"YoanHristov/xraynpc{part}" for part in range(1, 6)}
WEIGHT_FILE = {
    1: "resnet18.pt",
    2: "resnet18.pt",
    3: "resnet18.pt",
    4: "resnet18.pt",
    5: "resnet18_xray3.pt"
}

# ---------------- Bounded Cache ----------------
class BoundedCache(OrderedDict):
    """Thread-safe bounded FIFO cache preventing unbounded memory growth."""
    def __init__(self, maxsize: int = 250):
        super().__init__()
        self.maxsize = max(1, maxsize)
        self._lock = threading.Lock()

    def get_item(self, key):
        with self._lock:
            return super().get(key)

    def set_item(self, key, value):
        with self._lock:
            if key not in self and len(self) >= self.maxsize:
                self.popitem(last=False)
            super().__setitem__(key, value)

    def size(self):
        with self._lock:
            return len(self)

    def clear(self):
        with self._lock:
            super().clear()

gemini_cache = BoundedCache(maxsize=250)
gemini_client = None
registry = {}


# ---------------- Model & Weight Resolution ----------------
def fetch(part: int, name: str) -> Path:
    search_dirs = [
        Path("outputs") / ("model_repo" if part == 5 else f"part_models/part{part}"),
        Path("../VeritasAIResearch/outputs") / ("model_repo" if part == 5 else f"part_models/part{part}"),
        Path("C:/Users/User/Desktop/VeritasAIResearch/outputs") / ("model_repo" if part == 5 else f"part_models/part{part}")
    ]
    for folder in search_dirs:
        candidate = folder / name
        if candidate.exists():
            return candidate

    logger.info(f"Local file not found for Part {part} ({name}). Downloading from Hugging Face Hub: {HF_REPOS[part]}...")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPOS[part], filename=name))


def new_net(num_classes: int = 3) -> nn.Module:
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net.eval()
    return net


def load_model(part: int) -> dict:
    cfg_path = fetch(part, "config.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    classes = cfg.get("classes", CLASSES[:2] if part in (1, 2) else CLASSES)
    net = new_net(len(classes))
    weight_name = WEIGHT_FILE[part]
    weight_path = fetch(part, weight_name)
    state = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return {"cfg": cfg, "net": net}


def load_registry():
    global gemini_client
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            gemini_client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
            logger.info("Google GenAI client initialized successfully.")
        except Exception as err:
            logger.warning(f"Failed to initialize Google GenAI client: {err}")
            gemini_client = None
    else:
        logger.warning("CRITICAL: GEMINI_API_KEY missing! Gemini second reader running in single-reader provisional mode.")

    for part in range(1, 6):
        try:
            registry[part] = load_model(part)
            logger.info(f"Loaded Model {part}: {NAMES[part]}")
        except Exception as err:
            logger.error(f"Failed to load Model {part}: {err}", exc_info=True)


# ---------------- Preprocessing ----------------
def load_gray(source) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            image.draft("L", (IMG, IMG))
            return np.asarray(image.convert("L").resize((IMG, IMG), Image.BILINEAR))
    except Exception as error:
        raise ValueError("Provided file is not a valid or readable image.") from error


def raw_input(img: np.ndarray) -> torch.Tensor:
    return (torch.tensor(img, dtype=torch.float32).div(255).expand(3, -1, -1) - MEAN) / STD


def standardize(img: np.ndarray, fill: float, target_std: float) -> np.ndarray:
    x = img.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6) * target_std + fill
    return np.clip(x, 0, 255).astype(np.uint8)


def to_input(img: np.ndarray, fill: float, target_std: float) -> torch.Tensor:
    x = torch.tensor(standardize(img, fill, target_std), dtype=torch.float32).div(255).expand(3, -1, -1)
    return (x - MEAN) / STD


def zscore(img: np.ndarray) -> torch.Tensor:
    x = torch.tensor(img, dtype=torch.float32)
    return ((x - x.mean()) / (x.std() + 1e-6)).unsqueeze(0).expand(3, -1, -1)


def model_input(part: int, cfg: dict, gray: np.ndarray) -> torch.Tensor:
    if part == 1:
        return raw_input(gray)
    if part == 4:
        return zscore(gray)
    fill = cfg.get("fill", 123)
    target_std = cfg.get("target_std", 56.47)
    return to_input(gray, fill, target_std)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# ---------------- Safe Grad-CAM & Heatmap Colormap ----------------
def gradcam(net: nn.Module, x: torch.Tensor):
    saved = {}
    hook = net.layer4.register_forward_hook(lambda module, inp, out: saved.update(maps=out))
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = net(x)
            top_class = logits[0].argmax()
            grads = torch.autograd.grad(logits[0, top_class], saved["maps"])[0]
            heat = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * saved["maps"]).sum(dim=1)).squeeze(0).detach().cpu().numpy()
            max_val = heat.max()
            norm_heat = heat / max_val if max_val > 0 else heat
            return norm_heat, logits.detach().cpu().numpy()[0]
    finally:
        hook.remove()


def apply_colormap_jet(gray_uint8: np.ndarray) -> np.ndarray:
    """Fast vectorized colormap without matplotlib (saves ~100MB RAM)."""
    x = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(x * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(x * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(x * 4.0 - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def overlay_heatmap(orig_gray: np.ndarray, heat_norm: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blends colorized Grad-CAM overlay onto the grayscale X-ray."""
    orig_rgb = np.stack([orig_gray, orig_gray, orig_gray], axis=-1)
    heat_rgb = apply_colormap_jet((heat_norm * 255).astype(np.uint8))
    orig_pil = Image.fromarray(orig_rgb)
    heat_pil = Image.fromarray(heat_rgb).resize(orig_pil.size, Image.BILINEAR)
    return Image.blend(orig_pil, heat_pil, alpha=alpha)


# ---------------- Gemini Second Reader ----------------
GEM_QUESTIONS = {
    "xray": {"type": "noul", "instructions": "Is this image a chest X-ray?"},
    "quality": {
        "type": "choice",
        "instructions": "Rate the technical quality of this chest X-ray for diagnosis. Ignore disease findings.",
        "criteria": {
            "High": "Sharp, well exposed, lungs fully visible.",
            "Noisy": "Usable, but visible noise, low contrast, wrong exposure, rotation or clutter (text, devices).",
            "Unacceptable": "Not diagnostic: heavy blur, lungs cut off, not a chest X-ray, or strong artifacts."
        }
    },
    "diagnosis": {
        "type": "choice",
        "instructions": "Which of these is shown on this chest X-ray? Judge only from the image.",
        "criteria": {
            "NORMAL": "Clear lungs, no sign of pneumonia or COVID-19.",
            "PNEUMONIA": "Bacterial or viral pneumonia that is NOT COVID-19.",
            "COVID-19": "Pattern typical for COVID-19: bilateral, peripheral ground-glass opacities."
        }
    }
}


def ask_gemini(image_bytes: bytes, model_name: str = "gemini-decision-1") -> dict:
    key = hashlib.sha256(image_bytes).hexdigest()
    cached = gemini_cache.get_item(key)
    if cached is not None:
        return cached
    if gemini_client is None:
        return None

    buffer = io.BytesIO()
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("L")
        img.thumbnail((GEM_SIDE, GEM_SIDE))
        img.save(buffer, "PNG")

    try:
        response = gemini_client.decisions.create(
            model=model_name,
            questions=GEM_QUESTIONS,
            input=[{
                "type": "image",
                "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                "mime_type": "image/png",
                "resolution": "high"
            }]
        )
        answers = response.answers
        out = {
            "xray": float(answers["xray"].noul),
            "quality": str(answers["quality"].choice),
            "quality_conf": float(answers["quality"].confidence),
            "cls": str(answers["diagnosis"].choice),
            "cls_conf": float(answers["diagnosis"].confidence)
        }
        gemini_cache.set_item(key, out)
        return out
    except Exception as error:
        logger.warning(f"Gemini API unavailable: {str(error)[:120]}")
        return None


# ---------------- Clinical Decision Engine ----------------
def decide(p: np.ndarray, g: dict, cfg: dict) -> dict:
    classes = cfg.get("classes", CLASSES)
    r = int(p.argmax())
    conf = float(p.max())
    temp = cfg.get("temperature", 1.0)
    t_res = cfg.get("t_res", 0.40)
    t_gem = cfg.get("t_gem", 0.80)
    t_covid = cfg.get("t_covid", 0.85)
    scope_min = cfg.get("scope_min", 0.50)
    quality_conf = cfg.get("quality_conf", 0.70)
    quality_blocks = cfg.get("quality_blocks", False)
    covid_override = cfg.get("covid_override", True)

    ev = [f"ResNet-18: {classes[r]} at {conf:.0%} (threshold {t_res:.0%}, calibrated T={temp:.2f})"]
    v = {
        "tier": "",
        "cls": None,
        "risk": "NONE",
        "headline": "",
        "action": "",
        "banner": None,
        "agree": "N/A",
        "evidence": ev
    }

    if g is None:
        v.update(tier="U", banner="Scope not verified")
        ev.append("Gemini unavailable: image type, quality, and second reader check are missing.")
        if conf >= t_res:
            v["risk"] = "LOW" if r == 0 else "HIGH"
            v["headline"] = f"PROVISIONAL {v['risk']} RISK: {classes[r]} (one reader)"
        else:
            v["risk"] = "REVIEW"
            v["headline"] = "INCONCLUSIVE: ResNet below its confidence threshold"
        v["action"] = "Single-reader result. Check yourself that this is a usable chest X-ray, and treat the answer as provisional."
        return v

    in_scope = float(g["xray"]) >= scope_min
    bad = g["quality"] == "Unacceptable" and float(g["quality_conf"]) >= quality_conf

    ev.insert(0, f"Chest X-ray? Gemini P = {float(g['xray']):.0%} (rule: at least {scope_min:.0%}). Quality: {g['quality']} at {float(g['quality_conf']):.0%}.")
    ev.append(f"Gemini Decision 1: {g['cls']} at {float(g['cls_conf']):.0%} (threshold {t_gem:.0%})")

    if not in_scope or (quality_blocks and bad):
        v.update(
            tier="X",
            risk="NONE",
            headline="NO READING: this does not look like a chest X-ray" if not in_scope else "NO READING: image quality is not diagnostic",
            action="Upload a frontal chest radiograph." if not in_scope else "Retake or re-export the image."
        )
        return v

    gc = classes.index(g["cls"]) if g["cls"] in classes else -1
    v["agree"] = "AGREE" if r == gc else "DISAGREE"

    if r == gc:
        problems = [
            m for m, b in [
                ("the ResNet is below its threshold", conf < t_res),
                ("Gemini is below its threshold", float(g["cls_conf"]) < t_gem),
                (f"image quality is {g['quality']}", g["quality"] != "High")
            ] if b
        ]
        v.update(tier="B" if problems else "A", cls=r, risk="LOW" if r == 0 else "HIGH")
        v["headline"] = "LOW RISK: no urgent flag" if r == 0 else f"HIGH RISK: {classes[r]}, review first"
        v["action"] = (
            "Routine queue. A normal screen does not exclude disease: act on symptoms. "
            if r == 0 else "Two independent readers agree. Review first. "
        ) + (f"Caution: {'; '.join(problems)}." if problems else "")

    elif covid_override and r == 2 and len(p) > 2 and p[2] >= t_covid:
        v.update(
            tier="C",
            cls=2,
            risk="HIGH",
            headline="HIGH RISK: COVID-19 suspected (ResNet-led)",
            action=f"Gemini read {g['cls']}, but ResNet is {p[2]:.0%} sure of COVID-19 (override threshold {t_covid:.0%}). Review first and confirm."
        )

    else:
        flagged = (r > 0) + (gc > 0)
        reason = (
            "both readers suspect disease but disagree on the type" if flagged == 2
            else "one reader suspects disease, the other says NORMAL" if flagged == 1
            else "the readers disagree"
        )
        v.update(
            tier="D",
            risk="REVIEW",
            headline=f"{'URGENT REVIEW: ' if flagged else 'REVIEW: '}{reason}",
            action="Human review." + (" Disease is suspected by at least one reader: do not file as routine." if flagged else "")
        )

    return v


# ---------------- 8-Stage Visual Stepper Orchestrator ----------------
def analyze_model(reg: dict, part: int, image_bytes: bytes) -> dict:
    total_start = time.perf_counter()
    stages = []
    cfg = reg[part]["cfg"]
    net = reg[part]["net"]
    classes = cfg.get("classes", CLASSES[:2] if part in (1, 2) else CLASSES)

    # Stage 1: Upload & Resize
    t0 = time.perf_counter()
    gray = load_gray(image_bytes)
    t1 = time.perf_counter()
    stages.append({
        "id": "upload",
        "label": "Upload & Resize (224x224 Gray)",
        "status": "done",
        "latency_ms": round((t1 - t0) * 1000, 2),
        "detail": "Converted to canonical 224x224 grayscale"
    })

    # Stage 2: Scope & Quality Audit (Gemini)
    t_scope_start = time.perf_counter()
    gemini_out = None
    if part == 5:
        gemini_model_name = cfg.get("gemini_model", "gemini-decision-1")
        gemini_out = ask_gemini(image_bytes, model_name=gemini_model_name)
        t_scope_end = time.perf_counter()
        if gemini_out is not None:
            stages.append({
                "id": "scope_check",
                "label": "Scope & Quality Audit (Gemini)",
                "status": "done",
                "latency_ms": round((t_scope_end - t_scope_start) * 1000, 2),
                "detail": f"X-ray prob: {gemini_out['xray']:.0%} | Quality: {gemini_out['quality']}"
            })
        else:
            stages.append({
                "id": "scope_check",
                "label": "Scope & Quality Audit (Gemini)",
                "status": "failed",
                "latency_ms": round((t_scope_end - t_scope_start) * 1000, 2),
                "detail": "Gemini API unavailable / key missing"
            })
    else:
        stages.append({
            "id": "scope_check",
            "label": "Scope & Quality Audit (Gemini)",
            "status": "skipped",
            "latency_ms": 0.0,
            "detail": "Not active in Part I–IV baseline"
        })

    # Out-of-Scope (Tier X) Early Circuit Breaker
    if part == 5 and gemini_out is not None:
        scope_min = cfg.get("scope_min", 0.50)
        quality_conf = cfg.get("quality_conf", 0.70)
        quality_blocks = cfg.get("quality_blocks", False)
        in_scope = gemini_out["xray"] >= scope_min
        bad_quality = gemini_out["quality"] == "Unacceptable" and gemini_out["quality_conf"] >= quality_conf
        if not in_scope or (quality_blocks and bad_quality):
            stages.extend([
                {"id": "preprocess", "label": "Standardization", "status": "skipped", "latency_ms": 0.0, "detail": "Halted: Out of scope"},
                {"id": "resnet", "label": "ResNet-18 Backbone", "status": "skipped", "latency_ms": 0.0, "detail": "Halted: Out of scope"},
                {"id": "calibration", "label": "Temperature Calibration", "status": "skipped", "latency_ms": 0.0, "detail": "Halted: Out of scope"},
                {"id": "gradcam", "label": "Grad-CAM Attribution", "status": "skipped", "latency_ms": 0.0, "detail": "Halted: Out of scope"},
                {"id": "gemini_reader", "label": "Gemini Second Reader", "status": "skipped", "latency_ms": 0.0, "detail": "Halted: Out of scope"},
                {"id": "decision", "label": "Consensus Decision Rule", "status": "done", "latency_ms": 0.1, "detail": "Tier X: Reject out-of-scope image"}
            ])
            total_latency = round((time.perf_counter() - total_start) * 1000, 2)
            return {
                "model": 5,
                "name": NAMES[5],
                "tier": "X",
                "risk": "NONE",
                "headline": "NO READING: this does not look like a chest X-ray" if not in_scope else "NO READING: image quality is not diagnostic",
                "action": "Upload a frontal chest radiograph." if not in_scope else "Retake or re-export the image.",
                "banner": "Out of scope image detected",
                "agree": "N/A",
                "evidence": [f"Gemini chest X-ray probability {gemini_out['xray']:.0%} below threshold {scope_min:.0%}"],
                "stages": stages,
                "resnet": {"probabilities": {}, "heatmap_png_base64": ""},
                "gemini": gemini_out,
                "total_latency_ms": total_latency,
                "frozen_on_validation": {k: cfg[k] for k in ("temperature", "t_res", "t_covid", "t_gem", "scope_min", "quality_conf", "validation") if k in cfg},
                "disclaimer": "Decision support for a clinician. Not a diagnosis.",
                "_raw_gray": gray,
                "_raw_heat": np.zeros((IMG, IMG), dtype=np.float32)
            }

    # Stage 3: Preprocessing
    t2 = time.perf_counter()
    tensor_input = model_input(part, cfg, gray).unsqueeze(0).to(DEVICE)
    t3 = time.perf_counter()
    stages.append({
        "id": "preprocess",
        "label": "Standardization",
        "status": "done",
        "latency_ms": round((t3 - t2) * 1000, 2),
        "detail": PREPROCESS_DESCRIPTIONS.get(part, "Standardization")
    })

    # Stage 4 & 6: ResNet Backbone & Grad-CAM
    t4 = time.perf_counter()
    heat, logits = gradcam(net, tensor_input)
    t5 = time.perf_counter()
    stages.append({
        "id": "resnet",
        "label": "ResNet-18 Backbone",
        "status": "done",
        "latency_ms": round((t5 - t4) * 1000, 2),
        "detail": f"Forward pass on CPU ({len(classes)} classes)"
    })

    # Stage 5: Calibration
    t6 = time.perf_counter()
    if part == 5:
        temp = cfg.get("temperature", 1.0)
        p = softmax(logits[None], temperature=temp)[0]
        t7 = time.perf_counter()
        stages.append({
            "id": "calibration",
            "label": "Temperature Calibration",
            "status": "done",
            "latency_ms": round((t7 - t6) * 1000, 2),
            "detail": f"Scaled by T={temp:.2f}"
        })
    else:
        p = softmax(logits[None], temperature=1.0)[0]
        stages.append({
            "id": "calibration",
            "label": "Temperature Calibration",
            "status": "skipped",
            "latency_ms": 0.0,
            "detail": "Not active in Part I–IV baseline"
        })

    # Heatmap Base64
    t8 = time.perf_counter()
    heat_png = io.BytesIO()
    Image.fromarray((heat * 255).astype(np.uint8)).resize((IMG, IMG), Image.BILINEAR).save(heat_png, "PNG")
    heat_b64 = base64.b64encode(heat_png.getvalue()).decode("utf-8")
    t9 = time.perf_counter()
    stages.append({
        "id": "gradcam",
        "label": "Grad-CAM Attribution",
        "status": "done",
        "latency_ms": round((t9 - t8) * 1000, 2),
        "detail": "Layer 4 spatial saliency extracted (224x224 intensity)"
    })

    # Stage 7: Gemini Second Reader
    if part == 5:
        stages.append({
            "id": "gemini_reader",
            "label": "Gemini Second Reader",
            "status": "done" if gemini_out is not None else "failed",
            "latency_ms": 0.0,
            "detail": "Cached / Parallel from scope check" if gemini_out is not None else "Gemini unavailable"
        })
    else:
        stages.append({
            "id": "gemini_reader",
            "label": "Gemini Second Reader",
            "status": "skipped",
            "latency_ms": 0.0,
            "detail": "Not active in Part I–IV baseline"
        })

    # Stage 8: Decision Engine
    t10 = time.perf_counter()
    if part == 5:
        decision = decide(p, gemini_out, cfg)
    else:
        r = int(p.argmax())
        conf = float(p.max())
        triage_conf = cfg.get("triage_conf", 0.70)
        risk = ("LOW" if r == 0 else "HIGH") if conf >= triage_conf else "REVIEW"
        headline = f"{risk} RISK: {classes[r]}" if risk != "REVIEW" else "INCONCLUSIVE: confidence below threshold"
        decision = {
            "tier": "",
            "risk": risk,
            "headline": headline,
            "action": "Decision support for a clinician. Uncalibrated single model: no second reader, no image scope check.",
            "banner": None,
            "agree": "N/A",
            "evidence": [f"{NAMES[part]}: {classes[r]} at {conf:.0%} (badge threshold {triage_conf:.0%}, probabilities not calibrated)"]
        }
    t11 = time.perf_counter()

    stages.append({
        "id": "decision",
        "label": "Consensus Decision Rule",
        "status": "done",
        "latency_ms": round((t11 - t10) * 1000, 2),
        "detail": f"Tier {decision['tier']}: {decision['risk']} Risk" if decision['tier'] else f"{decision['risk']} Risk"
    })

    total_latency = round((time.perf_counter() - total_start) * 1000, 2)

    return {
        "model": part,
        "name": NAMES[part],
        "tier": decision["tier"],
        "risk": decision["risk"],
        "headline": decision["headline"],
        "action": decision["action"],
        "banner": decision["banner"],
        "agree": decision["agree"],
        "evidence": decision["evidence"],
        "stages": stages,
        "resnet": {
            "probabilities": {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, p)},
            "heatmap_png_base64": heat_b64
        },
        "gemini": gemini_out,
        "total_latency_ms": total_latency,
        "frozen_on_validation": {k: cfg[k] for k in ("temperature", "t_res", "t_covid", "t_gem", "scope_min", "quality_conf", "validation") if k in cfg},
        "disclaimer": "Decision support for a clinician. Not a diagnosis.",
        "_raw_gray": gray,
        "_raw_heat": heat
    }


# ---------------- FastAPI Application Setup ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Veritas AI model registry and configurations...")
    load_registry()
    logger.info("Veritas AI backend initialized successfully. Ready for incoming traffic.")
    yield
    registry.clear()
    gemini_cache.clear()
    logger.info("Veritas AI backend successfully shut down.")


app = FastAPI(
    title="Veritas AI: Chest X-ray Consensus Cascade",
    description="Stand-alone, high-speed backend for 5 ResNet-18 chest radiograph models.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------- REST API Endpoints ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "models_loaded": sorted(list(registry.keys())),
        "gemini_configured": gemini_client is not None,
        "cache_size": gemini_cache.size(),
        "device": str(DEVICE)
    }


@app.get("/models")
def list_models():
    return [
        {
            "id": p,
            "name": NAMES[p],
            "classes": registry[p]["cfg"].get("classes", CLASSES[:2] if p in (1, 2) else CLASSES),
            "description": registry[p]["cfg"].get("description", DESCRIPTION_5 if p == 5 else f"Part {p} Model"),
            "stages": [
                {"id": s_id, "label": s_lbl, "active": s_id in ACTIVE_STAGES[p]}
                for s_id, s_lbl in STAGE_DEFINITIONS
            ]
        }
        for p in sorted(registry.keys())
    ]


@app.post("/analyze/{model_id}")
async def analyze_model_endpoint(model_id: int, file: UploadFile = File(...)):
    if model_id not in registry:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found. Must be between 1 and 5.")

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Image size exceeds maximum limit of {MAX_BYTES // (1024 * 1024)} MB.")

    try:
        res = await asyncio.to_thread(analyze_model, registry, model_id, content)
        # Clean private helper arrays before returning pure JSON
        clean_res = {k: v for k, v in res.items() if not k.startswith("_")}
        return clean_res
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except Exception as err:
        logger.error(f"Inference error on Model {model_id}: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during image inference.")


@app.post("/analyze")
async def analyze_default_endpoint(file: UploadFile = File(...)):
    return await analyze_model_endpoint(model_id=5, file=file)


# ---------------- Interactive Gradio UI ----------------
TIER_COLORS = {
    "A": "#10b981",  # Emerald Green
    "B": "#f59e0b",  # Amber
    "C": "#ef4444",  # Crimson Red
    "D": "#f97316",  # Orange
    "X": "#6b7280",  # Gray
    "U": "#64748b"   # Slate
}

MODEL_NAME_TO_ID = {
    "Part V: Consensus Cascade (Recommended)": 5,
    "Part I: Baseline ResNet-18": 1,
    "Part II: Shortcut-Robust": 2,
    "Part III: + COVID-19 (3-Class)": 3,
    "Part IV: Gemini-Filtered Training": 4
}


def gradio_predict(image_pil, model_choice):
    if image_pil is None:
        return (
            "### ⚠️ Please upload a chest radiograph image to begin.",
            None,
            {},
            [],
            "No input image provided."
        )

    # Make sure registry is initialized (handles standalone Gradio launch)
    if not registry:
        load_registry()

    model_id = MODEL_NAME_TO_ID.get(model_choice, 5)

    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    raw_bytes = buffer.getvalue()

    result = analyze_model(registry, model_id, raw_bytes)

    tier = result.get("tier", "")
    risk = result.get("risk", "NONE")
    headline = result.get("headline", "")
    action = result.get("action", "")
    banner = result.get("banner")
    tier_color = TIER_COLORS.get(tier, "#3b82f6")

    # 1. Format Markdown Diagnostic Badge
    tier_badge_html = f"""
    <div style="background-color: {tier_color}18; border-left: 5px solid {tier_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 1.25rem; font-weight: 800; color: {tier_color};">
                {f"TIER {tier} • " if tier else ""}{risk} RISK
            </span>
            <span style="font-size: 0.85rem; color: #94a3b8; font-family: monospace;">
                Latency: {result['total_latency_ms']} ms
            </span>
        </div>
        <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">
            {headline}
        </div>
        {f'<div style="background: #f43f5e22; color: #f43f5e; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; margin-top: 6px; display: inline-block;">⚠️ {banner}</div>' if banner else ''}
    </div>
    """

    # 2. Grad-CAM Overlay Image
    raw_gray = result.get("_raw_gray")
    raw_heat = result.get("_raw_heat")
    if raw_gray is not None and raw_heat is not None and raw_heat.max() > 0:
        overlay_pil = overlay_heatmap(raw_gray, raw_heat, alpha=0.45)
    else:
        overlay_pil = image_pil.resize((IMG, IMG))

    # 3. Probabilities Label
    probs = result.get("resnet", {}).get("probabilities", {})

    # 4. Latency Stepper Audit Table
    stages_clean = [
        {
            "Stage": s["label"],
            "Status": s["status"].upper(),
            "Latency (ms)": s["latency_ms"],
            "Detail": s.get("detail") or ""
        }
        for s in result.get("stages", [])
    ]

    # 5. Clinical Evidence & Action
    evidence_items = "\n".join([f"- {e}" for e in result.get("evidence", [])])
    evidence_md = f"""
### 📋 Clinical Recommendation
**Recommended Action**: {action}

#### Reader Findings & Audit Trail:
{evidence_items}

*Disclaimer: {result.get('disclaimer')}*
    """

    return tier_badge_html, overlay_pil, probs, stages_clean, evidence_md


# Construct Gradio Interface with custom modern medical theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="Veritas AI: Chest Radiograph Consensus Cascade", theme=theme) as demo:
    gr.Markdown(
        """
        # 🫁 Veritas AI: Chest Radiograph Consensus Cascade
        **Clinical Second-Reader Diagnostic Support Engine**  
        *Combines ResNet-18 Deep Attribution with Gemini Decision 1 second-reader verification, automated scope/quality auditing, and temperature-calibrated risk tiers.*
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            input_image = gr.Image(type="pil", label="Upload Frontal Chest Radiograph", height=320)
            model_selector = gr.Dropdown(
                choices=list(MODEL_NAME_TO_ID.keys()),
                value="Part V: Consensus Cascade (Recommended)",
                label="Select Diagnostic Model"
            )
            submit_btn = gr.Button("🔬 Run Consensus Analysis", variant="primary", size="lg")

            gr.Markdown(
                """
                > **Tip for Judges / Clinicians**:
                > - Upload any chest radiograph to evaluate concordance.
                > - Upload a non-chest photo (e.g. coffee cup or pet) to see **Tier X Out-of-Scope Circuit Breaker** reject it instantly!
                """
            )

        with gr.Column(scale=6):
            output_badge = gr.HTML(label="Diagnostic Assessment")

            with gr.Row():
                with gr.Column(scale=5):
                    output_cam = gr.Image(type="pil", label="Grad-CAM Spatial Saliency (Layer 4)", height=260)
                with gr.Column(scale=5):
                    output_probs = gr.Label(label="Calibrated Class Probabilities", num_top_classes=3)

            output_evidence = gr.Markdown(label="Evidence & Recommendations")

    with gr.Accordion("⏱️ 8-Stage Visual Stepper & Latency Audit", open=False):
        output_stages = gr.JSON(label="Pipeline Latency Breakdown (ms)")

    submit_btn.click(
        fn=gradio_predict,
        inputs=[input_image, model_selector],
        outputs=[output_badge, output_cam, output_probs, output_stages, output_evidence]
    )

# Mount Gradio onto the FastAPI application at root ("/")
# ssr_mode=False: on HF Spaces, Gradio 5.x/6.x enables SSR by default, which starts a
# Node proxy on GRADIO_SERVER_PORT (7860) and makes our uvicorn.run() fail with
# "[Errno 98] address already in use".
app = gr.mount_gradio_app(app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import uvicorn
    server_port = int(
        os.getenv("GRADIO_SERVER_PORT", "7860")
        if os.getenv("SPACE_ID")
        else os.getenv("PORT", "7860")
    )
    uvicorn.run(app, host="0.0.0.0", port=server_port)

