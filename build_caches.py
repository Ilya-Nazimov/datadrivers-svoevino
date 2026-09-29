#!/usr/bin/env python3
"""
Precompute all gallery caches before building Docker image.
Run this once, then copy the caches into the Docker image.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Import everything from app.py
from app import (
    ServiceState, GalleryViewDataset, SigLIPGalleryDataset,
    load_bottle_bgr, label_band, sift_pack,
    _normalize_text, char_ngrams, crop_label_region, run_ocr_single,
    IMAGES_DIR, IMAGES_CSV, WINE_CSV, DINO_CHECKPOINT,
    SIGLIP_IMG_CHECKPOINT, SIGLIP_TEXT_CHECKPOINT, SIGLIP_MODEL_NAME,
    KEYPOINT_CACHE, EMB_CACHE, GALLERY_VIEWS,
    SIFT_NFEATURES_DB, OCR_NGRAM_SIZES, OCR_LABEL_T0, OCR_LABEL_T1, OCR_MAX_SIZE,
    BICUBIC, IMAGENET_MEAN, IMAGENET_STD,
)


def make_clahe():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def build_sift_cache(gallery_paths, cache_dir, nfeatures, clahe=False, force=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta = [p.name for p in gallery_paths]
    if not force and meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            print("SIFT keypoint cache is up-to-date")
            return meta
    print(f"Building SIFT keypoint cache for {len(meta)} images (clahe={clahe})...")
    cl = make_clahe() if clahe else None
    t0 = time.time()
    for i, p in enumerate(gallery_paths):
        bgr = load_bottle_bgr(p)
        if bgr is None:
            pts = np.zeros((0, 2), np.float32)
            desc = np.zeros((0, 128), np.uint8)
        else:
            gray = cv2.cvtColor(label_band(bgr), cv2.COLOR_BGR2GRAY)
            pts, desc = sift_pack(gray, nfeatures, 0.03, cl)
        np.savez_compressed(cache_dir / f"{i:05d}.npz", pts=pts, desc=desc)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(meta)} ({time.time() - t0:.0f}s)")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"SIFT cache built in {time.time() - t0:.0f}s")
    return meta


def build_ocr_cache(gallery_paths, cache_dir, engine_type="easyocr", force=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"gallery_ocr_{engine_type}_v1.json"

    cache = {}
    if cache_path.exists() and not force:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

    to_process = [p for p in gallery_paths if p.name not in cache]
    if not to_process:
        print(f"OCR cache HIT ({len(gallery_paths)} images)")
        return cache

    print(f"OCR ({engine_type}): {len(to_process)} new images to process")

    if engine_type == "easyocr":
        import easyocr
        reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
    else:
        raise ValueError(f"Unsupported engine: {engine_type}")

    t0 = time.time()
    for i, p in enumerate(to_process):
        try:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            label_img = crop_label_region(img, OCR_LABEL_T0, OCR_LABEL_T1, OCR_MAX_SIZE)
            text = run_ocr_single(reader, engine_type, label_img)
            cache[p.name] = text
        except Exception as e:
            print(f"  OCR failed for {p.name}: {e}")
            cache[p.name] = ""

        if (i + 1) % 10 == 0 or i + 1 == len(to_process):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"  OCR {i + 1}/{len(to_process)} ({elapsed:.0f}s, ~{eta:.0f}s left)")

        if (i + 1) % 50 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"OCR cache saved: {cache_path.name}")
    return cache


def main():
    print("=" * 70)
    print("BUILDING ALL CACHES")
    print("=" * 70)

    # Load gallery
    gdf = pd.read_csv(IMAGES_CSV)
    gdf["img"] = gdf["img"].astype(str)
    gallery_paths = []
    for _, row in gdf.iterrows():
        pth = IMAGES_DIR / row["img"]
        if pth.exists():
            gallery_paths.append(pth)
    print(f"Gallery: {len(gallery_paths)} images")

    # 1. SIFT keypoints
    print("\n[1/2] Building SIFT keypoints cache...")
    build_sift_cache(gallery_paths, KEYPOINT_CACHE, SIFT_NFEATURES_DB, clahe=False)

    # 2. OCR texts
    print("\n[2/2] Building OCR gallery cache...")
    build_ocr_cache(gallery_paths, Path("ocr_cache"), engine_type="easyocr")

    print("\n" + "=" * 70)
    print("ALL CACHES BUILT")
    print("=" * 70)
    print("Now you can build the Docker image:")
    print("  docker build -t wine-service .")


if __name__ == "__main__":
    main()