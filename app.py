#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wine Retrieval Service v5.5 — single-file FastAPI server.
Точное повторение логики из экспериментального скрипта evaluate_quadro_ocr.py.
"""

import hashlib  # <-- ДОБАВЛЕНО: нужен для _meta_hash
import io
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

# DINOv2 вызывает upsample_bicubic2d, которого на MPS нет: без этого флага пересчёт
# галереи на Mac падает с NotImplementedError ("Application startup failed"). Флаг надо
# выставить ДО первого вызова torch, поэтому он здесь, рядом с импортами.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ====================================================================== #
#  Константы
# ====================================================================== #
try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GALLERY_VIEWS = ["whole", "label"]

# Версии кэшей (должны совпадать с экспериментальным скриптом v5.5)
GALLERY_VERSION = 1
SIGLIP_VERSION = 1
SIGLIP_TEXT_VERSION = 1

# ========== Веса и параметры из v5.5 ==========
W_GEO = 0.07203409069213
W_DINO = 5.73561463991306
W_SIGLIP = 103.46855397613823
W_SIGLIP_TEXT = 0.5416
W_OCR = 33.7596
TEXT_TOP_K = 15
OCR_TOP_K = 11
TIEBREAK_GAP = 0.0174
MIN_INLIERS = 12
CANDIDATES = 17

# Максимальная высота входного фото; всё крупнее ужимается с сохранением пропорций.
MAX_QUERY_HEIGHT = 1500

# SIFT params
SIFT_RATIO = 0.6868944956304118
SIFT_THRESH = 10.722485199651263
SIFT_GUIDED = False  # --no-guided
SIFT_GUIDED_RADIUS = 5.575240791198452
SIFT_NFEATURES_DB = 1500
SIFT_NFEATURES_Q = 1500
SIFT_CONTRAST_Q = 0.01843646645664742

# OCR params
OCR_LABEL_T0 = 0.01
OCR_LABEL_T1 = 0.99
OCR_MAX_SIZE = 1000
OCR_NGRAM_SIZES = (3, 4)

# Paths
IMAGES_DIR = Path("images")
IMAGES_CSV = Path("images/images_match.csv")
WINE_CSV = Path("wine.csv")
DINO_CHECKPOINT = Path("runs/wine_v2/best.pt")
SIGLIP_IMG_CHECKPOINT = Path("runs/siglip_enhanced_old/siglip_enhanced_last.pth")
SIGLIP_TEXT_CHECKPOINT = Path("runs/siglip_reranker/final_model.pth")
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
YOLO_MODEL = "yolov8x-worldv2.pt"
KEYPOINT_CACHE = Path("keypoint_cache")
EMB_CACHE = Path("emb_cache")


# ====================================================================== #
#  Cache key helpers (должны совпадать с v5.5 для подхвата кэшей)
# ====================================================================== #
def _meta_hash(meta):
    s = json.dumps(meta, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _ckpt_key(checkpoint_path):
    p = Path(checkpoint_path).resolve()
    st = p.stat()
    # Только имя и РАЗМЕР, без mtime. mtime меняется при каждом git clone / checkout
    # (в т.ч. при вытягивании из Git LFS), из-за чего хэш кэша эмбеддингов переставал
    # совпадать и галерея из 4218 фото пересчитывалась при каждом запуске — это минуты
    # ожидания вместо секунд и прямой риск сорваться на таймаут валидации.
    # Размер + имя — достаточная защита от подмены чекпоинта: другое содержимое даёт
    # другой размер, а полное хеширование 2.3 ГБ на каждом старте слишком дорого.
    return f"{p.name}|{st.st_size}"


# ====================================================================== #
#  Device
# ====================================================================== #
def get_device():
    # WINE_FORCE_CPU=1 — принудительно CPU. Нужно на Apple Silicon: DINOv2 зовёт
    # upsample_bicubic2d, которого на MPS нет, и с PYTORCH_ENABLE_MPS_FALLBACK каждый
    # такой вызов уходит на CPU через fallback — медленнее, чем честный CPU, плюс
    # лишние предупреждения. Пересчёт галереи на MPS-конфигурации особенно медленный.
    if os.getenv("WINE_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


# ====================================================================== #
#  DINOv2 Models
# ====================================================================== #
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.30, easy_margin=False):
        super().__init__()
        self.s, self.m, self.easy_margin = s, m, easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels=None):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, w)
        if labels is None:
            return self.s * cosine, x
        bs = labels.size(0)
        tgt = cosine[torch.arange(bs, device=labels.device), labels]
        sine = torch.sqrt((1.0 - tgt.pow(2)).clamp(min=0.0))
        phi = tgt * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(tgt > 0.0, phi, tgt)
        else:
            phi = torch.where(tgt > self.th, phi, tgt - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi.unsqueeze(1) + (1.0 - one_hot) * cosine
        return self.s * logits, x


class DINOv2ArcFace(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, proj_dim=512, dropout=0.1,
                 s=30.0, m=0.30, easy_margin=False):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, proj_dim, bias=False))
        nn.init.trunc_normal_(self.proj[2].weight, std=0.02)
        self.head = ArcFaceHead(proj_dim, num_classes, s=s, m=m, easy_margin=easy_margin)

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", feats.get("x_norm", list(feats.values())[0]))
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if feats.dim() == 3:
            feats = feats[:, 0]
        return self.head(self.proj(feats), labels)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p)))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class DINOv2ArcFaceV2(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, proj_dim=512, dropout=0.1,
                 s=30.0, m=0.35, pooling="cat", easy_margin=False):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.gem = GeM()
        feat_dim = embed_dim * 2 if pooling == "cat" else embed_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, proj_dim, bias=False))
        nn.init.trunc_normal_(self.proj[2].weight, std=0.02)
        self.head = ArcFaceHead(proj_dim, num_classes, s=s, m=m, easy_margin=easy_margin)

    def forward(self, x, labels=None):
        try:
            out = self.backbone(x, is_training=True)
        except TypeError:
            out = self.backbone(x)
        if isinstance(out, dict):
            cls, pat = out.get("x_norm_clstoken"), out.get("x_norm_patchtokens")
        else:
            cls, pat = out, None
        if cls is not None and cls.dim() == 3:
            cls = cls[:, 0]
        if self.pooling == "cls" or pat is None:
            f = cls
        elif self.pooling == "gem":
            f = self.gem(pat)
        else:
            f = torch.cat([cls, self.gem(pat)], dim=1)
        return self.head(self.proj(f), labels)


# ====================================================================== #
#  SigLIP Model
# ====================================================================== #
class SigLIPEncoder(nn.Module):
    def __init__(self, siglip_model):
        super().__init__()
        self.siglip = siglip_model

    def _extract_image_features(self, pixel_values):
        image_output = self.siglip.get_image_features(pixel_values=pixel_values)
        if isinstance(image_output, torch.Tensor):
            return image_output
        elif hasattr(image_output, 'pooler_output') and image_output.pooler_output is not None:
            return image_output.pooler_output
        elif hasattr(image_output, 'last_hidden_state'):
            return image_output.last_hidden_state[:, 0]
        else:
            vision_output = self.siglip.vision_model(pixel_values=pixel_values)
            if hasattr(vision_output, 'pooler_output') and vision_output.pooler_output is not None:
                return vision_output.pooler_output
            return vision_output[0][:, 0]

    def forward(self, pixel_values):
        features = self._extract_image_features(pixel_values)
        return F.normalize(features, p=2, dim=1)


# ====================================================================== #
#  OCR
# ====================================================================== #
def _normalize_text(text):
    if text is None:
        return ""
    s = str(text).lower().replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def char_ngrams(text, sizes=OCR_NGRAM_SIZES):
    s = _normalize_text(text)
    counter = Counter()
    if not s:
        return counter
    for n in sizes:
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            counter[s[i:i + n]] += 1
    return counter


def ngram_dice(a, b):
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    if inter == 0:
        return 0.0
    total = sum(a.values()) + sum(b.values())
    return 2.0 * inter / total


def crop_label_region(img_pil, t0=0.05, t1=0.95, max_size=1000):
    w, h = img_pil.size
    top = int(h * t0)
    bottom = int(h * t1)
    cropped = img_pil.crop((0, top, w, bottom))
    cw, ch = cropped.size
    max_dim = max(cw, ch)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        cropped = cropped.resize((new_w, new_h), BICUBIC)
    return cropped


def _preprocess_for_ocr(img_pil):
    img_np = np.array(img_pil)
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return result


def run_ocr_single(ocr_engine, engine_type, img_pil):
    if ocr_engine is None:
        return ""
    img_processed = _preprocess_for_ocr(img_pil)
    if engine_type == "easyocr":
        try:
            results = ocr_engine.readtext(img_processed, detail=1, paragraph=False)
            lines = []
            for item in results:
                if len(item) >= 3:
                    bbox, text, conf = item[0], item[1], item[2]
                    if conf > 0.1:
                        lines.append(str(text))
                elif len(item) >= 2:
                    lines.append(str(item[1]))
            return " ".join(lines)
        except Exception:
            return ""
    elif engine_type == "paddleocr":
        try:
            result = ocr_engine.ocr(img_processed, cls=True)
        except Exception:
            try:
                result = ocr_engine.ocr(img_processed)
            except Exception:
                return ""
        lines = []
        if result is None:
            return ""
        for page in result:
            if page is None:
                continue
            if isinstance(page, list):
                for item in page:
                    try:
                        if isinstance(item, list) and len(item) >= 2:
                            inner = item[1]
                            if isinstance(inner, (list, tuple)) and len(inner) >= 1:
                                lines.append(str(inner[0]))
                            else:
                                lines.append(str(inner))
                        elif isinstance(item, dict) and 'text' in item:
                            lines.append(str(item['text']))
                    except Exception:
                        continue
        return " ".join(lines)
    return ""


# ====================================================================== #
#  YOLO
# ====================================================================== #
def detect_and_crop_bottle_pil(img_pil, yolo_model, prompt="wine bottle",
                               conf_threshold=0.1, padding_pct=0.02):
    try:
        img_w, img_h = img_pil.size
        img_cx, img_cy = img_w / 2, img_h / 2
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img_pil.save(tmp.name, quality=95)
            tmp_path = tmp.name
        results = yolo_model(tmp_path, verbose=False, conf=conf_threshold, imgsz=640)
        os.unlink(tmp_path)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                box_cx = (x1 + x2) / 2
                box_cy = (y1 + y2) / 2
                dist = math.sqrt(((box_cx - img_cx) / img_w) ** 2 +
                                 ((box_cy - img_cy) / img_h) ** 2)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'dist_to_center': dist,
                })

        if not detections:
            return img_pil, False

        detections.sort(key=lambda d: (d['dist_to_center'], -d['conf']))
        best = detections[0]

        x1, y1, x2, y2 = best['bbox']
        w = x2 - x1
        h = y2 - y1
        pad_x = w * padding_pct
        pad_y = h * padding_pct

        x1_crop = max(0, int(x1 - pad_x))
        y1_crop = max(0, int(y1 - pad_y))
        x2_crop = min(img_w, int(x2 + pad_x))
        y2_crop = min(img_h, int(y2 + pad_y))

        cropped = img_pil.crop((x1_crop, y1_crop, x2_crop, y2_crop))
        return cropped, True

    except Exception:
        return img_pil, False


# ====================================================================== #
#  Datasets
# ====================================================================== #
def load_bottle(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    bbox = img.split()[-1].getbbox()
    if bbox is None:
        return img
    pad = int(0.01 * max(img.size))
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - pad), max(0, y0 - pad),
                     min(img.width, x1 + pad), min(img.height, y1 + pad)))


def _paste_on_gray(bottle, size, height_frac=0.85):
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    w, h = bottle.size
    if w <= 0 or h <= 0:
        return canvas, (0, 0, 0, 0)
    aspect = w / h
    th = int(size * height_frac)
    tw = int(th * aspect)
    if tw > int(size * 0.96):
        tw = int(size * 0.96); th = int(tw / aspect)
    if th > int(size * 0.98):
        th = int(size * 0.98); tw = int(th / aspect)
    b = bottle.resize((tw, th), BICUBIC)
    left, top = (size - tw) // 2, (size - th) // 2
    canvas.paste(b, (left, top), b)
    return canvas.convert("RGB"), (left, top, tw, th)


class GalleryViewDataset(Dataset):
    def __init__(self, image_paths, image_size, view):
        self.image_paths, self.image_size, self.view = image_paths, image_size, view
        self.transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        bottle = load_bottle(self.image_paths[idx])
        if self.view == "label":
            img, (left, top, tw, th) = _paste_on_gray(bottle, self.image_size)
            if tw > 0 and th > 0:
                bt, bb = top + int(th * 0.40), top + int(th * 0.88)
                bl = max(0, left - int(tw * 0.05))
                br = min(self.image_size, left + tw + int(tw * 0.05))
                img = img.crop((bl, bt, br, bb)).resize(
                    (self.image_size, self.image_size), BICUBIC)
            else:
                img, _ = _paste_on_gray(bottle, self.image_size)
        else:
            img, _ = _paste_on_gray(bottle, self.image_size)
        return self.transform(img)


class SigLIPGalleryDataset(Dataset):
    def __init__(self, image_paths, image_size):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def query_multicrop_views(img_pil, size):
    w, h = img_pil.size
    def ssc(sz):
        if w < h:
            nw, nh = sz, int(h * sz / w)
        else:
            nh, nw = sz, int(w * sz / h)
        r = img_pil.resize((nw, nh), BICUBIC)
        l, t = (nw - sz) // 2, (nh - sz) // 2
        return r.crop((l, t, l + sz, t + sz))
    def cc(fw, fh, cy=0.5):
        cw, ch = max(1, min(int(w * fw), w)), max(1, min(int(h * fh), h))
        l = max(0, min(int(w / 2 - cw / 2), w - cw))
        t = max(0, min(int(h * cy - ch / 2), h - ch))
        return img_pil.crop((l, t, l + cw, t + ch)).resize((size, size), BICUBIC)
    return [ssc(size), cc(0.7, 0.7, 0.50), cc(0.5, 0.5, 0.55),
            cc(0.9, 0.45, 0.55), cc(0.35, 0.35, 0.55), cc(0.9, 0.30, 0.42)]


def siglip_multicrop_views(img_pil):
    """3 crops для SigLIP (как в SigLIPQueryDataset multicrop=True)."""
    w, h = img_pil.size
    def center_crop(fw, fh, cy=0.5):
        cw = max(1, min(int(w * fw), w))
        ch = max(1, min(int(h * fh), h))
        l = max(0, min(int(w / 2 - cw / 2), w - cw))
        t = max(0, min(int(h * cy - ch / 2), h - ch))
        return img_pil.crop((l, t, l + cw, t + ch))
    return [img_pil, center_crop(0.8, 0.8, 0.50), center_crop(0.7, 0.7, 0.50)]


# ====================================================================== #
#  SIFT Keypoint side
# ====================================================================== #
def load_bottle_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        x, y, w, h = cv2.boundingRect((alpha > 8).astype(np.uint8))
        if w <= 0 or h <= 0:
            return None
        pad = int(0.01 * max(img.shape[:2]))
        return img[max(0, y - pad):min(img.shape[0], y + h + pad),
               max(0, x - pad):min(img.shape[1], x + w + pad), :3]
    return img[:, :, :3]


def label_band(bgr, t0=0.40, t1=0.88):
    h = bgr.shape[0]
    return bgr[int(h * t0):int(h * t1), :]


def center_crop(bgr, fw, fh, cy=0.5):
    H, W = bgr.shape[:2]
    cw, ch = max(1, min(int(W * fw), W)), max(1, min(int(H * fh), H))
    l = max(0, min(int(W / 2 - cw / 2), W - cw))
    t = max(0, min(int(H / 2 - ch / 2), H - ch))
    return bgr[t:t + ch, l:l + cw]


def sift_pack(gray, nfeatures, contrast=0.03, clahe_obj=None):
    if clahe_obj is not None:
        gray = clahe_obj.apply(gray)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast)
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(kp) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.array([k.pt for k in kp], dtype=np.float32)
    return pts, np.clip(np.round(desc), 0, 255).astype(np.uint8)


def rootsift(d):
    f = d.astype(np.float32) + 1e-7
    f /= f.sum(axis=1, keepdims=True)
    return np.sqrt(f)


def verify_pair(qpts, qroot, cpts, croot, ratio, thresh):
    """SIFT verification без guided (--no-guided)."""
    if len(qpts) == 0 or len(cpts) == 0:
        return 0, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    mm = bf.knnMatch(qroot, croot, k=2)
    good = []
    for pair in mm:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append((pair[0].queryIdx, pair[0].trainIdx))
    if len(good) < 6:
        return 0, 0, len(good)
    qi = np.array([g[0] for g in good], dtype=np.int64)
    ti = np.array([g[1] for g in good], dtype=np.int64)
    src = cpts[ti].reshape(-1, 1, 2).astype(np.float32)
    dst = qpts[qi].reshape(-1, 1, 2).astype(np.float32)

    inl_h, M_h = 0, None
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, thresh,
                                 maxIters=5000, confidence=0.999)
    if mask is not None:
        inl_h = int(mask.sum())
        M_h = (H, True)
    inl_a, M_a = 0, None
    A, mask2 = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=thresh,
                                           maxIters=5000, confidence=0.999)
    if mask2 is not None:
        inl_a = int(mask2.sum())
        M_a = (A, False)

    if inl_h >= inl_a:
        return inl_h, inl_h, 0
    else:
        return inl_a, inl_a, 0


# ====================================================================== #
#  Service State
# ====================================================================== #
class ServiceState:
    def __init__(self):
        self.device = get_device()
        self.gallery_paths = []
        self.gallery_labels = []
        self.N = 0

        self.dino_model = None
        self.dino_image_size = 224
        self.dino_gallery = None  # [V, N, D]

        self.yolo_model = None

        self.siglip_img_model = None
        self.siglip_img_size = 224  # будет перезаписан из конфига чекпоинта
        self.siglip_img_gallery = None  # [N, D]
        self.siglip_img_transform = None  # создается после загрузки модели

        self.siglip_text_model = None
        self.siglip_text_size = 224  # будет перезаписан из конфига чекпоинта
        self.siglip_text_gallery = None  # [N, D]
        self.siglip_text_processor = None
        self.siglip_text_transform = None  # создается после загрузки модели

        self.sift_db_pts = []
        self.sift_db_desc = []

        self.ocr_engine = None
        self.ocr_engine_type = None
        self.gallery_ocr_texts = {}  # {filename: text}
        self.gallery_ocr_ngrams = {}  # {filename: Counter}

        self.dino_transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def load_gallery(self):
        gdf = pd.read_csv(IMAGES_CSV)
        gdf["img"] = gdf["img"].astype(str)
        for _, row in gdf.iterrows():
            pth = IMAGES_DIR / row["img"]
            if pth.exists():
                self.gallery_paths.append(pth)
                self.gallery_labels.append(row["label"])
        self.N = len(self.gallery_paths)
        print(f"[INIT] Gallery: {self.N} images")

    def load_yolo(self):
        try:
            from ultralytics import YOLO
            print(f"[INIT] Loading YOLO: {YOLO_MODEL}")
            self.yolo_model = YOLO(YOLO_MODEL)
            self.yolo_model.set_classes(["wine bottle"])
        except Exception as e:
            print(f"[INIT] YOLO unavailable: {e}")
            self.yolo_model = None

    def load_dino(self):
        print(f"[INIT] Loading DINOv2 from {DINO_CHECKPOINT}")
        ckpt = torch.load(DINO_CHECKPOINT, map_location="cpu", weights_only=False)
        train_args = ckpt.get("args", {})
        self.dino_image_size = int(train_args.get("image_size", 224))

        state = ckpt["model"]
        num_classes, proj_dim = state["head.weight"].shape
        is_v2 = "gem.p" in state
        pooling = train_args.get("pooling", "cat") if is_v2 else "cls"
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        embed_dim = getattr(backbone, "embed_dim", 768)
        cls_kw = dict(dropout=train_args.get("dropout", 0.1),
                      s=train_args.get("arcface_s", 30.0),
                      m=train_args.get("arcface_m", 0.30),
                      easy_margin=train_args.get("easy_margin", False))
        if is_v2:
            model = DINOv2ArcFaceV2(backbone, embed_dim, num_classes, proj_dim,
                                    pooling=pooling, **cls_kw)
        else:
            model = DINOv2ArcFace(backbone, embed_dim, num_classes, proj_dim, **cls_kw)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.dino_model = model
        print(f"[INIT] DINOv2 loaded: arch={'v2' if is_v2 else 'v1'}, "
              f"pooling={pooling}, size={self.dino_image_size}")

    # ---------------------------------------------------------------- #
    #  DINO gallery: с проверкой кэша
    # ---------------------------------------------------------------- #
    def compute_dino_gallery(self):
        emb_dir = EMB_CACHE
        emb_dir.mkdir(parents=True, exist_ok=True)
        ckpt_key = _ckpt_key(DINO_CHECKPOINT)
        meta = {
            "kind": "gallery", "version": GALLERY_VERSION, "ckpt": ckpt_key,
            "image_size": self.dino_image_size, "views": GALLERY_VIEWS,
            "files": [p.name for p in self.gallery_paths],
        }
        path = emb_dir / f"gallery_{_meta_hash(meta)}.pt"
        if path.exists():
            print(f"[INIT] DINO gallery cache HIT: {path.name}")
            self.dino_gallery = torch.load(path, map_location="cpu", weights_only=False)["Gv"]
            print(f"[INIT] DINO gallery: {self.dino_gallery.shape}")
            return

        print("[INIT] Computing DINO gallery embeddings...")
        embs = []
        for view in GALLERY_VIEWS:
            ds = GalleryViewDataset(self.gallery_paths, self.dino_image_size, view)
            dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
            view_embs = []
            with torch.inference_mode():
                for imgs in dl:
                    _, e = self.dino_model(imgs.to(self.device), None)
                    view_embs.append(F.normalize(e, dim=1).cpu())
            embs.append(torch.cat(view_embs, dim=0))
        self.dino_gallery = torch.stack(embs)
        torch.save({"meta": meta, "Gv": self.dino_gallery}, path)
        print(f"[INIT] DINO gallery saved: {path.name}, shape={self.dino_gallery.shape}")

    def load_siglip_img(self):
        from transformers import AutoProcessor, AutoModel
        print(f"[INIT] Loading SigLIP img from {SIGLIP_IMG_CHECKPOINT}")
        ckpt = torch.load(SIGLIP_IMG_CHECKPOINT, map_location="cpu", weights_only=False)
        config = ckpt.get('config', {})
        model_name = config.get('model_name', SIGLIP_MODEL_NAME)
        # ВАЖНО: размер берём строго из конфига чекпоинта, не из аргументов
        self.siglip_img_size = config.get('img_size', 224)

        backbone = AutoModel.from_pretrained(model_name)
        model = SigLIPEncoder(backbone)
        state = {k.replace('siglip.', ''): v for k, v in ckpt['model_state_dict'].items()
                 if k.startswith('siglip.')}
        model.siglip.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        self.siglip_img_model = model

        # Создаём трансформ под реальный размер модели
        self.siglip_img_transform = T.Compose([
            T.Resize((self.siglip_img_size, self.siglip_img_size), interpolation=BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"[INIT] SigLIP img loaded: size={self.siglip_img_size} (from checkpoint config)")

    # ---------------------------------------------------------------- #
    #  SigLIP img gallery: с проверкой кэша
    # ---------------------------------------------------------------- #
    def compute_siglip_img_gallery(self):
        emb_dir = EMB_CACHE
        emb_dir.mkdir(parents=True, exist_ok=True)
        ckpt_key = _ckpt_key(SIGLIP_IMG_CHECKPOINT)
        meta = {
            "kind": "siglip_img_gallery", "version": SIGLIP_VERSION,
            "ckpt": ckpt_key, "image_size": self.siglip_img_size,
            "files": [p.name for p in self.gallery_paths],
        }
        path = emb_dir / f"siglip_img_gallery_{_meta_hash(meta)}.pt"
        if path.exists():
            print(f"[INIT] SigLIP img gallery cache HIT: {path.name}")
            self.siglip_img_gallery = torch.load(path, map_location="cpu", weights_only=False)["embeds"]
            print(f"[INIT] SigLIP img gallery: {self.siglip_img_gallery.shape}")
            return

        print("[INIT] Computing SigLIP img gallery embeddings...")
        ds = SigLIPGalleryDataset(self.gallery_paths, self.siglip_img_size)
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        out = []
        with torch.inference_mode():
            for batch in dl:
                batch = batch.to(self.device)
                embeds = self.siglip_img_model(batch)
                out.append(embeds.cpu())
        self.siglip_img_gallery = torch.cat(out, dim=0)
        torch.save({"meta": meta, "embeds": self.siglip_img_gallery}, path)
        print(f"[INIT] SigLIP img gallery saved: {path.name}, shape={self.siglip_img_gallery.shape}")

    def load_siglip_text(self):
        from transformers import AutoProcessor, AutoModel
        if not SIGLIP_TEXT_CHECKPOINT.exists() or not WINE_CSV.exists():
            print("[INIT] SigLIP text: checkpoint or wine.csv not found, skipping")
            return
        print(f"[INIT] Loading SigLIP text from {SIGLIP_TEXT_CHECKPOINT}")
        ckpt = torch.load(SIGLIP_TEXT_CHECKPOINT, map_location="cpu", weights_only=False)
        config = ckpt.get('config', {})
        model_name = config.get('model_name', SIGLIP_MODEL_NAME)
        # ВАЖНО: размер берём строго из конфига чекпоинта
        self.siglip_text_size = config.get('img_size', 224)
        projection_dim = config.get('projection_dim', 512)

        self.siglip_text_processor = AutoProcessor.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)

        try:
            from siglip_negative_train import SigLIPReranker
            model = SigLIPReranker(
                base_model=backbone,
                projection_dim=projection_dim,
                freeze_base=True
            )
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"[INIT] SigLIP text: projection heads loaded (dim={projection_dim})")
        except ImportError:
            model = SigLIPEncoder(backbone)
            state = {k.replace('siglip.', ''): v
                     for k, v in ckpt['model_state_dict'].items()
                     if k.startswith('siglip.')}
            if state:
                model.siglip.load_state_dict(state, strict=False)
            print("[INIT] SigLIP text: standard SigLIP fallback")

        model.to(self.device)
        model.eval()
        self.siglip_text_model = model

        # Создаём трансформ под реальный размер модели
        self.siglip_text_transform = T.Compose([
            T.Resize((self.siglip_text_size, self.siglip_text_size), interpolation=BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"[INIT] SigLIP text loaded: size={self.siglip_text_size} (from checkpoint config)")

    # ---------------------------------------------------------------- #
    #  SigLIP text gallery: с проверкой кэша
    # ---------------------------------------------------------------- #
    def compute_siglip_text_gallery(self):
        if self.siglip_text_model is None:
            return
        wine_df = pd.read_csv(WINE_CSV)
        text_map = {}
        for _, row in wine_df.iterrows():
            key_label = str(row.get("label", "")).strip()
            key_img = str(row.get("img", "")).strip()
            parts = []
            for col in ['name', 'producer', 'grape', 'category_color']:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    parts.append(str(val).strip())
            text_str = ", ".join(parts) if parts else "wine bottle"
            if key_label:
                text_map[key_label] = text_str
            if key_img:
                text_map[key_img] = text_str

        gallery_texts = []
        for lab, pth in zip(self.gallery_labels, self.gallery_paths):
            t = text_map.get(str(lab).strip())
            if t is None:
                t = text_map.get(pth.name)
            gallery_texts.append(t if t else "wine bottle")

        # Проверка кэша: имя файла как в v5.5
        emb_dir = EMB_CACHE
        emb_dir.mkdir(parents=True, exist_ok=True)
        text_hash = hashlib.md5("".join(gallery_texts).encode("utf-8")).hexdigest()[:16]
        ckpt_key = _ckpt_key(SIGLIP_TEXT_CHECKPOINT)
        has_proj = hasattr(self.siglip_text_model, 'text_projection')
        proj_suffix = "_proj" if has_proj else "_base"
        path = emb_dir / f"siglip_text_{ckpt_key}_{text_hash}{proj_suffix}.pt"
        if path.exists():
            print(f"[INIT] SigLIP text gallery cache HIT: {path.name}")
            self.siglip_text_gallery = torch.load(path, map_location="cpu", weights_only=False)["embeds"]
            print(f"[INIT] SigLIP text gallery: {self.siglip_text_gallery.shape}")
            return

        print("[INIT] Computing SigLIP text gallery embeddings...")
        out = []
        with torch.inference_mode():
            for i in range(0, len(gallery_texts), 8):
                batch_texts = gallery_texts[i:i+8]
                inputs = self.siglip_text_processor(
                    text=batch_texts, padding="longest",
                    truncation=True, max_length=64, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if hasattr(self.siglip_text_model, 'get_text_features') and hasattr(self.siglip_text_model, 'text_projection'):
                    feats = self.siglip_text_model.get_text_features(**inputs)
                elif hasattr(self.siglip_text_model, 'get_text_features'):
                    feats = self.siglip_text_model.get_text_features(**inputs)
                    if not isinstance(feats, torch.Tensor):
                        if hasattr(feats, 'text_embeds'):
                            feats = feats.text_embeds
                        elif hasattr(feats, 'pooler_output'):
                            feats = feats.pooler_output
                        elif hasattr(feats, 'last_hidden_state'):
                            feats = feats.last_hidden_state[:, 0]
                    feats = F.normalize(feats, p=2, dim=1)
                elif hasattr(self.siglip_text_model, 'text_model'):
                    feats = self.siglip_text_model.text_model(**inputs)
                    if hasattr(feats, 'pooler_output'):
                        feats = feats.pooler_output
                    feats = F.normalize(feats, p=2, dim=1)
                else:
                    feats = self.siglip_text_model(**inputs)
                    if hasattr(feats, 'pooler_output'):
                        feats = feats.pooler_output
                    feats = F.normalize(feats, p=2, dim=1)
                out.append(feats.cpu())
        self.siglip_text_gallery = torch.cat(out, dim=0)
        torch.save({"embeds": self.siglip_text_gallery, "dim": self.siglip_text_gallery.shape[1]}, path)
        print(f"[INIT] SigLIP text gallery saved: {path.name}, shape={self.siglip_text_gallery.shape}")

    def load_sift_cache(self):
        print(f"[INIT] Loading SIFT keypoints from {KEYPOINT_CACHE}")
        meta_path = KEYPOINT_CACHE / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"SIFT cache not found at {KEYPOINT_CACHE}. Run build_caches.py first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.sift_db_pts = []
        self.sift_db_desc = []
        for i in range(len(meta)):
            z = np.load(KEYPOINT_CACHE / f"{i:05d}.npz")
            self.sift_db_pts.append(z["pts"])
            self.sift_db_desc.append(z["desc"])
        print(f"[INIT] SIFT keypoints loaded: {len(meta)} images")

    def load_ocr(self):
        print("[INIT] Loading EasyOCR...")
        import easyocr
        self.ocr_engine = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        self.ocr_engine_type = "easyocr"

        ocr_cache_path = Path("ocr_cache") / f"gallery_ocr_{self.ocr_engine_type}_v1.json"
        if not ocr_cache_path.exists():
            raise RuntimeError(f"OCR cache not found at {ocr_cache_path}. Run build_caches.py first.")
        self.gallery_ocr_texts = json.loads(ocr_cache_path.read_text(encoding="utf-8"))
        self.gallery_ocr_ngrams = {
            name: char_ngrams(text, OCR_NGRAM_SIZES)
            for name, text in self.gallery_ocr_texts.items()
        }
        print(f"[INIT] OCR cache loaded: {len(self.gallery_ocr_texts)} texts")

    def self_check(self):
        """Dummy forward каждой модели на её размере — падает сразу на старте
        при любом несовпадении размеров, а не на первом запросе."""
        print("[INIT] Running self-check (dummy forward on each model)...")
        with torch.inference_mode():
            # DINO
            dummy = torch.zeros(1, 3, self.dino_image_size, self.dino_image_size).to(self.device)
            _ = self.dino_model(dummy, None)
            del dummy

            # SigLIP img
            dummy = torch.zeros(1, 3, self.siglip_img_size, self.siglip_img_size).to(self.device)
            _ = self.siglip_img_model(dummy)
            del dummy

            # SigLIP text
            if self.siglip_text_model is not None:
                dummy = torch.zeros(1, 3, self.siglip_text_size, self.siglip_text_size).to(self.device)
                out = self.siglip_text_model(dummy)
                if isinstance(out, tuple):
                    _ = out[0]
                del dummy

        print(f"[INIT] Self-check passed: DINO={self.dino_image_size}, "
              f"SigLIP_img={self.siglip_img_size}, SigLIP_text={self.siglip_text_size}")


# ====================================================================== #
#  Inference logic (exact v5.5 reproduction)
# ====================================================================== #
def run_inference(state: ServiceState, img_pil: Image.Image) -> str:
    t0_total = time.time()
    # Замер по стадиям: печатается в конце вместе с суммарным временем
    _t = {}

    # Stage 0: YOLO crop
    if state.yolo_model is not None:
        query_img, _ = detect_and_crop_bottle_pil(img_pil, state.yolo_model)
    else:
        query_img = img_pil
    _t['yolo'] = time.time() - t0_total

    # Stage 1: DINOv2 query embeddings (6 views)
    views = query_multicrop_views(query_img, state.dino_image_size)
    batch = torch.stack([state.dino_transform(v) for v in views]).to(state.device)
    with torch.inference_mode():
        _, e = state.dino_model(batch, None)
        Qc = F.normalize(e, dim=1).cpu()  # [C, D]

    Gv = state.dino_gallery  # [V, N, D]
    Cn, Dn = Qc.shape
    Vn, Nn, _ = Gv.shape
    with torch.inference_mode():
        flat = Qc.reshape(Cn, Dn) @ Gv.reshape(Vn * Nn, Dn).t()
        sims = flat.reshape(Cn, Vn, Nn).permute(1, 0, 2)
        item_sim = sims.amax(dim=(0, 1))  # max fusion
    dino_scores = item_sim.numpy()
    _t['dino'] = time.time() - t0_total

    # Top-K candidates
    K = min(CANDIDATES, Nn)
    cand_idx = np.argsort(-dino_scores)[:K].tolist()

    # Stage 2: SIFT verification
    query_bgr = cv2.cvtColor(np.array(query_img), cv2.COLOR_RGB2BGR)
    sift_views = []
    for v in (query_bgr,
              center_crop(query_bgr, 0.8, 0.8, 0.50),
              center_crop(query_bgr, 0.9, 0.35, 0.55),
              center_crop(query_bgr, 0.6, 0.4, 0.30)):
        g = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        pts, desc = sift_pack(g, SIFT_NFEATURES_Q, SIFT_CONTRAST_Q)
        sift_views.append((pts, rootsift(desc)))

    geo = {c: -1 for c in range(Nn)}
    for c in cand_idx:
        cpts, cdesc = state.sift_db_pts[c], state.sift_db_desc[c]
        croot = rootsift(cdesc)
        best = (0, 0, 0)
        for qpts, qroot in sift_views:
            r = verify_pair(qpts, qroot, cpts, croot, SIFT_RATIO, SIFT_THRESH)
            if r[0] > best[0]:
                best = r
        geo[c] = best[0]
    _t['sift'] = time.time() - t0_total

    # Stage 1: Hybrid (DINO + SIFT)
    def key_hyb(c):
        g = geo[c]
        s = float(dino_scores[c])
        verified = 1 if g >= MIN_INLIERS else 0
        if verified:
            hybrid_score = g * W_GEO + s * W_DINO
            return (-1, -hybrid_score)
        else:
            return (0, -s)
    hyb_order = sorted(range(Nn), key=key_hyb)

    # Stage 3: SigLIP img — используем siglip_img_transform (размер из конфига чекпоинта)
    # Кропы SigLIP нужны дважды: в Stage 3 (img) и Stage 4 (text). Строятся из одного и
    # того же query_img, поэтому считаем их один раз — на CPU (MPS недоступен в контейнере)
    # повторное построение и стек трансформаций давали заметную долю латентности.
    _siglip_crops_cache = None

    def _siglip_crops():
        nonlocal _siglip_crops_cache
        if _siglip_crops_cache is None:
            _siglip_crops_cache = siglip_multicrop_views(query_img)
        return _siglip_crops_cache

    siglip_img_scores = None
    if state.siglip_img_model is not None:
        crops = _siglip_crops()
        batch = torch.stack([state.siglip_img_transform(c) for c in crops]).to(state.device)
        with torch.inference_mode():
            embeds_flat = state.siglip_img_model(batch)
            embeds = embeds_flat.mean(dim=0, keepdim=True)
            embeds = F.normalize(embeds, p=2, dim=1).cpu()
        with torch.inference_mode():
            siglip_img_sim = embeds @ state.siglip_img_gallery.T
        siglip_img_scores = siglip_img_sim.numpy()[0]
        _t['siglip_img'] = time.time() - t0_total

        def key_siglip_img(c):
            g = geo[c]
            s_dino = float(dino_scores[c])
            s_img = float(siglip_img_scores[c])
            verified = 1 if g >= MIN_INLIERS else 0
            if verified:
                return g * W_GEO + s_dino * W_DINO + s_img * W_SIGLIP
            else:
                return s_dino * W_DINO + s_img * W_SIGLIP
        siglip_img_order = sorted(range(Nn), key=key_siglip_img, reverse=True)
    else:
        siglip_img_order = hyb_order

    # Stage 4: SigLIP text reranking top-K — используем siglip_text_transform
    siglip_txt_order = siglip_img_order
    text_sims = None
    if state.siglip_text_model is not None and W_SIGLIP_TEXT > 0:
        crops = _siglip_crops()   # кэш из Stage 3, второй раз не строим
        batch = torch.stack([state.siglip_text_transform(c) for c in crops]).to(state.device)
        with torch.inference_mode():
            out = state.siglip_text_model(batch)
            if isinstance(out, tuple):
                out = out[0]
            emb = out.reshape(1, len(crops), -1).mean(dim=1)
            q_emb = F.normalize(emb, p=2, dim=1).cpu()
        with torch.inference_mode():
            text_sim = q_emb @ state.siglip_text_gallery.T
        text_sims = text_sim.numpy()[0]

        top_k_text = min(TEXT_TOP_K, len(siglip_img_order))
        top_candidates = siglip_img_order[:top_k_text]
        rest_candidates = siglip_img_order[top_k_text:]

        base_scores = {}
        for c in top_candidates:
            g = geo[c]
            s_dino = float(dino_scores[c])
            verified = 1 if g >= MIN_INLIERS else 0
            if siglip_img_scores is not None:
                s_img = float(siglip_img_scores[c])
                base_s = (g * W_GEO + s_dino * W_DINO + s_img * W_SIGLIP) if verified \
                    else (s_dino * W_DINO + s_img * W_SIGLIP)
            else:
                base_s = (g * W_GEO + s_dino * W_DINO) if verified \
                    else (s_dino * W_DINO)
            base_scores[c] = base_s

        # Strategy: "add"
        final_scores = {}
        for c in top_candidates:
            s_text = float(text_sims[c])
            final_scores[c] = base_scores[c] + s_text * W_SIGLIP_TEXT

        top_candidates.sort(key=lambda c: final_scores[c], reverse=True)
        siglip_txt_order = top_candidates + rest_candidates
    _t['siglip_text'] = time.time() - t0_total

    # Stage 5: OCR reranking top-N
    final_order = siglip_txt_order
    ocr_scores_for_query = {}

    if state.ocr_engine is not None and W_OCR > 0:
        top_k_ocr = min(OCR_TOP_K, len(siglip_txt_order))
        top_candidates_ocr = siglip_txt_order[:top_k_ocr]
        rest_candidates_ocr = siglip_txt_order[top_k_ocr:]

        label_img = crop_label_region(query_img, OCR_LABEL_T0, OCR_LABEL_T1, OCR_MAX_SIZE)
        query_ocr_text = run_ocr_single(state.ocr_engine, state.ocr_engine_type, label_img)
        query_ngrams = char_ngrams(query_ocr_text, OCR_NGRAM_SIZES) if query_ocr_text.strip() else Counter()

        if query_ngrams:
            base_scores_ocr = {}
            for c in top_candidates_ocr:
                g = geo[c]
                s_dino = float(dino_scores[c])
                verified = 1 if g >= MIN_INLIERS else 0
                if siglip_img_scores is not None:
                    s_img = float(siglip_img_scores[c])
                    base_s = (g * W_GEO + s_dino * W_DINO + s_img * W_SIGLIP) if verified \
                        else (s_dino * W_DINO + s_img * W_SIGLIP)
                else:
                    base_s = (g * W_GEO + s_dino * W_DINO) if verified \
                        else (s_dino * W_DINO)
                if text_sims is not None and W_SIGLIP_TEXT > 0:
                    base_s += float(text_sims[c]) * W_SIGLIP_TEXT
                base_scores_ocr[c] = base_s

            final_scores_ocr = {}
            for c in top_candidates_ocr:
                gallery_ngrams = state.gallery_ocr_ngrams.get(state.gallery_paths[c].name)
                ocr_score = ngram_dice(query_ngrams, gallery_ngrams) if gallery_ngrams else 0.0
                ocr_scores_for_query[c] = ocr_score
                final_scores_ocr[c] = base_scores_ocr[c] + ocr_score * W_OCR

            top_candidates_ocr.sort(key=lambda c: final_scores_ocr[c], reverse=True)
            final_order = top_candidates_ocr + rest_candidates_ocr
        else:
            for c in top_candidates_ocr:
                ocr_scores_for_query[c] = 0.0

    # Stage 6: top-2 tiebreak by OCR
    if state.ocr_engine is not None and siglip_img_scores is not None and len(final_order) >= 2:
        c1, c2 = final_order[0], final_order[1]
        s_img1 = float(siglip_img_scores[c1])
        s_img2 = float(siglip_img_scores[c2])
        tiebreak_gap_value = abs(s_img1 - s_img2)

        if tiebreak_gap_value < TIEBREAK_GAP:
            ocr1 = float(ocr_scores_for_query.get(c1, 0.0))
            ocr2 = float(ocr_scores_for_query.get(c2, 0.0))
            if max(ocr1, ocr2) > 0.0 and ocr2 > ocr1:
                final_order[0], final_order[1] = c2, c1

    best = final_order[0]
    elapsed = time.time() - t0_total
    _t['total'] = elapsed
    prev = 0.0
    parts = []
    for k in ("yolo", "dino", "sift", "siglip_img", "siglip_text", "total"):
        parts.append(f"{k}={_t.get(k, 0.0) - prev:.2f}" if k != "total"
                     else f"{k}={_t.get('total', 0.0):.2f}")
        if k != "total":
            prev = _t.get(k, prev)
    print(f"[TIMING] " + " ".join(parts) + " ocr=" + f"{elapsed - _t.get('siglip_text', 0.0):.2f}", flush=True)
    print(f"[PREDICT] -> {state.gallery_labels[best]} ({elapsed:.2f}s)")
    return state.gallery_labels[best]


# ====================================================================== #
#  FastAPI App
# ====================================================================== #
state = ServiceState()
app = FastAPI(title="Wine Retrieval Service", version="5.5")


@app.on_event("startup")
async def startup():
    print("=" * 70)
    print("WINE RETRIEVAL SERVICE v5.5")
    print("=" * 70)
    state.load_gallery()
    state.load_yolo()
    state.load_dino()
    state.compute_dino_gallery()
    state.load_siglip_img()
    state.compute_siglip_img_gallery()
    state.load_siglip_text()
    state.compute_siglip_text_gallery()
    state.load_sift_cache()
    state.load_ocr()
    state.self_check()  # dummy forward каждой модели — падает сразу при mismatch
    print("=" * 70)
    print("SERVICE READY")
    print("=" * 70)


@app.get("/health")
async def health():
    return {"status": "ok", "gallery_size": state.N}


@app.post("/v1/eval/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image")

        img_pil = Image.open(io.BytesIO(contents))
        img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")

        # Уменьшаем крупные снимки: на CPU инференс стоимость растёт вместе с числом
        # пикселей, а для сопоставления по SIFT/SigLIP хватает 1500 px по высоте.
        # Пропорции сохраняем, ресайм только вниз.
        if img_pil.height > MAX_QUERY_HEIGHT:
            new_w = max(1, round(img_pil.width * MAX_QUERY_HEIGHT / img_pil.height))
            img_pil = img_pil.resize((new_w, MAX_QUERY_HEIGHT), Image.LANCZOS)

        slug = run_inference(state, img_pil)
        return {"slug": slug}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)