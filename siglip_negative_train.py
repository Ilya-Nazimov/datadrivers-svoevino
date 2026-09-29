#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SigLIP Reranker Training with Hard Negative Mining

Обучает SigLIP модель для реранкинга вин с использованием:
- Positive pairs: (image, text) для одного вина
- Hard negatives: 5 самых похожих вин (по эмбеддингам до обучения)
- Contrastive loss для vision и text simultaneously
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Install with: pip install transformers")


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


class WineDataset(Dataset):
    """Датасет вин с изображениями и текстовыми описаниями."""

    def __init__(self, image_dir: Path, wine_csv: Path, processor, image_size: int = 224,
                 mode: str = "train", hard_negatives_map: Dict = None):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.image_size = image_size
        self.mode = mode
        self.hard_negatives_map = hard_negatives_map or {}

        self.df = pd.read_csv(wine_csv)

        required_cols = ['label', 'img']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in wine.csv")

        self.text_cols = ['name', 'producer', 'grape', 'category_color']
        self.available_text_cols = [c for c in self.text_cols if c in self.df.columns]

        valid_rows = []
        for idx, row in self.df.iterrows():
            img_path = self.image_dir / row['img']
            if img_path.exists():
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Loaded {len(self.df)} wines with valid images")

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def _get_text_description(self, row):
        parts = []
        for col in self.available_text_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return ", ".join(parts) if parts else "wine bottle"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row['img']

        try:
            img = ImageOps.exif_transpose(Image.open(img_path)).convert('RGB')
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img_tensor = torch.zeros(3, self.image_size, self.image_size)

        text = self._get_text_description(row)

        if self.mode == "extract":
            return img_tensor, text, idx
        else:
            # ВСЕГДА возвращаем hard negative для консистентности батча
            img_name = row['img']
            hard_negs = self.hard_negatives_map.get(img_name, [])

            if hard_negs:
                neg_img_name = random.choice(hard_negs)
                neg_rows = self.df[self.df['img'] == neg_img_name]
                if len(neg_rows) > 0:
                    neg_text = self._get_text_description(neg_rows.iloc[0])
                else:
                    # Fallback: случайный другой сэмпл
                    rand_idx = random.randint(0, len(self.df) - 1)
                    neg_text = self._get_text_description(self.df.iloc[rand_idx])
            else:
                # Fallback: случайный другой сэмпл
                rand_idx = random.randint(0, len(self.df) - 1)
                neg_text = self._get_text_description(self.df.iloc[rand_idx])

            return img_tensor, text, neg_text, idx


def extract_embeddings(model, processor, dataset, device, batch_size=32):
    """Извлекает image и text embeddings для всего датасета."""
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_img_embeds = []
    all_text_embeds = []

    with torch.inference_mode():
        for img_tensors, texts, indices in tqdm(loader, desc="Extracting embeddings"):
            img_tensors = img_tensors.to(device)

            # Image embeddings
            img_embeds = model.get_image_features(pixel_values=img_tensors)
            if not isinstance(img_embeds, torch.Tensor):
                if hasattr(img_embeds, 'image_embeds'):
                    img_embeds = img_embeds.image_embeds
                elif hasattr(img_embeds, 'pooler_output'):
                    img_embeds = img_embeds.pooler_output
            img_embeds = F.normalize(img_embeds, p=2, dim=1)

            # Text embeddings
            text_inputs = processor(
                text=list(texts),
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            text_embeds = model.get_text_features(**text_inputs)
            if not isinstance(text_embeds, torch.Tensor):
                if hasattr(text_embeds, 'text_embeds'):
                    text_embeds = text_embeds.text_embeds
                elif hasattr(text_embeds, 'pooler_output'):
                    text_embeds = text_embeds.pooler_output
            text_embeds = F.normalize(text_embeds, p=2, dim=1)

            all_img_embeds.append(img_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())

    all_img_embeds = torch.cat(all_img_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    return all_img_embeds, all_text_embeds


def find_hard_negatives(img_embeds, k=5):
    """Находит k самых похожих изображений для каждого изображения."""
    print(f"Finding {k} hard negatives for each image...")

    sim_matrix = img_embeds @ img_embeds.T

    hard_negatives = {}
    for i in range(len(img_embeds)):
        sims = sim_matrix[i].clone()
        sims[i] = -1e9  # Исключаем себя

        top_k_indices = torch.topk(sims, k=k).indices.tolist()
        hard_negatives[i] = top_k_indices

    print(f"Found hard negatives for {len(hard_negatives)} images")
    return hard_negatives


class SigLIPReranker(nn.Module):
    """SigLIP модель с projection heads для обучения."""

    def __init__(self, base_model, projection_dim=512, freeze_base=True):
        super().__init__()
        self.base_model = base_model

        # Определяем размерности из конфигурации
        vision_hidden = base_model.config.vision_config.hidden_size
        text_hidden = base_model.config.text_config.hidden_size

        self.img_projection = nn.Sequential(
            nn.Linear(vision_hidden, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def get_image_features(self, pixel_values):
        img_features = self.base_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(img_features, torch.Tensor):
            if hasattr(img_features, 'image_embeds'):
                img_features = img_features.image_embeds
            elif hasattr(img_features, 'pooler_output'):
                img_features = img_features.pooler_output
            elif hasattr(img_features, 'last_hidden_state'):
                img_features = img_features.last_hidden_state[:, 0]

        img_features = self.img_projection(img_features)
        return F.normalize(img_features, p=2, dim=1)

    def get_text_features(self, **kwargs):
        text_features = self.base_model.get_text_features(**kwargs)
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, 'text_embeds'):
                text_features = text_features.text_embeds
            elif hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                text_features = text_features.last_hidden_state[:, 0]

        text_features = self.text_projection(text_features)
        return F.normalize(text_features, p=2, dim=1)

    def forward(self, pixel_values=None, **text_kwargs):
        if pixel_values is not None:
            return self.get_image_features(pixel_values)
        else:
            return self.get_text_features(**text_kwargs)


class ContrastiveLoss(nn.Module):
    """InfoNCE loss для contrastive learning."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_embeds, text_embeds, hard_neg_text_embeds=None):
        """
        Args:
            img_embeds: [B, D] image embeddings
            text_embeds: [B, D] positive text embeddings
            hard_neg_text_embeds: [B, D] hard negative text embeddings
        """
        B = img_embeds.shape[0]

        # Positive logits
        pos_logits = F.cosine_similarity(img_embeds, text_embeds, dim=1) / self.temperature

        # All logits: similarity с каждым текстом в батче
        all_logits = (img_embeds @ text_embeds.T) / self.temperature

        # Маска для исключения positive pairs (диагональ)
        mask = torch.eye(B, dtype=torch.bool, device=img_embeds.device)
        all_logits = all_logits.masked_fill(mask, -1e9)

        # Если есть hard negatives - добавляем их как дополнительные негативы
        if hard_neg_text_embeds is not None and hard_neg_text_embeds.shape[0] == B:
            hard_neg_logits = F.cosine_similarity(
                img_embeds, hard_neg_text_embeds, dim=1
            ) / self.temperature
            # Конкатенируем: [B, B] -> [B, B+1]
            all_logits = torch.cat([all_logits, hard_neg_logits.unsqueeze(1)], dim=1)
        elif hard_neg_text_embeds is not None and hard_neg_text_embeds.shape[0] != B:
            # Если размеры не совпадают, добавляем только для совпадающей части
            min_b = min(B, hard_neg_text_embeds.shape[0])
            hard_neg_logits = F.cosine_similarity(
                img_embeds[:min_b], hard_neg_text_embeds[:min_b], dim=1
            ) / self.temperature
            # Добавляем к логитам для совпадающей части
            all_logits[:min_b] = torch.cat([
                all_logits[:min_b], hard_neg_logits.unsqueeze(1)
            ], dim=1)

        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + logsumexp(all_logits)
        loss = (-pos_logits + torch.logsumexp(all_logits, dim=1)).mean()

        return loss


def train_epoch(model, loader, loss_fn, optimizer, device, epoch):
    """Обучение одну эпоху."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    # Получаем processor из датасета
    processor = loader.dataset.processor

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        img_tensors, texts, neg_texts, indices = batch
        img_tensors = img_tensors.to(device)

        # Image embeddings
        img_embeds = model.get_image_features(pixel_values=img_tensors)

        # Positive text embeddings
        text_inputs_dict = processor(
            text=list(texts),
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        text_inputs_dict = {k: v.to(device) for k, v in text_inputs_dict.items()}
        text_embeds = model.get_text_features(**text_inputs_dict)

        # Hard negative text embeddings
        # Теперь ВСЕГДА есть для каждого сэмпла
        hard_neg_text_embeds = None
        if neg_texts is not None and len(neg_texts) > 0:
            neg_inputs_dict = processor(
                text=list(neg_texts),
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            neg_inputs_dict = {k: v.to(device) for k, v in neg_inputs_dict.items()}
            hard_neg_text_embeds = model.get_text_features(**neg_inputs_dict)

        # Loss
        loss = loss_fn(img_embeds, text_embeds, hard_neg_text_embeds)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, dataset, device, batch_size=32):
    """Валидация: вычисляем recall@k."""
    model.eval()

    processor = dataset.processor
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_img_embeds = []
    all_text_embeds = []

    with torch.inference_mode():
        for img_tensors, texts, indices in tqdm(loader, desc="Validating"):
            img_tensors = img_tensors.to(device)

            img_embeds = model.get_image_features(pixel_values=img_tensors)

            text_inputs = processor(
                text=list(texts),
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_embeds = model.get_text_features(**text_inputs)

            all_img_embeds.append(img_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())

    all_img_embeds = torch.cat(all_img_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    # Similarity matrix
    sim_matrix = all_img_embeds @ all_text_embeds.T

    correct_top1 = 0
    correct_top5 = 0
    total = len(all_img_embeds)

    for i in range(total):
        sims = sim_matrix[i]
        top_k = torch.topk(sims, k=min(5, total)).indices

        if top_k[0] == i:
            correct_top1 += 1
        if i in top_k:
            correct_top5 += 1

    recall_top1 = correct_top1 / total
    recall_top5 = correct_top5 / total

    return recall_top1, recall_top5


def main():
    parser = argparse.ArgumentParser(description="Train SigLIP Reranker with Hard Negatives")

    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--wine-csv", type=str, default="wine.csv")
    parser.add_argument("--base-model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-hard-negatives", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="runs/siglip_reranker")
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading base model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = AutoModel.from_pretrained(args.base_model)

    # Датасет для извлечения эмбеддингов
    print("Creating dataset for embedding extraction...")
    dataset_extract = WineDataset(
        image_dir=args.image_dir,
        wine_csv=args.wine_csv,
        processor=processor,
        image_size=args.image_size,
        mode="extract"
    )

    # Извлекаем эмбеддинги
    print("Extracting embeddings with base model...")
    base_model = base_model.to(device)
    img_embeds, text_embeds = extract_embeddings(
        base_model, processor, dataset_extract, device, batch_size=args.batch_size
    )

    # Hard negatives
    print("Finding hard negatives...")
    hard_negatives_indices = find_hard_negatives(img_embeds, k=args.num_hard_negatives)

    # Преобразуем в имена файлов
    hard_negatives_map = {}
    for idx, neg_indices in hard_negatives_indices.items():
        img_name = dataset_extract.df.iloc[idx]['img']
        neg_names = [dataset_extract.df.iloc[neg_idx]['img'] for neg_idx in neg_indices]
        hard_negatives_map[img_name] = neg_names

    # Датасет для обучения
    print("Creating training dataset with hard negatives...")
    dataset_train = WineDataset(
        image_dir=args.image_dir,
        wine_csv=args.wine_csv,
        processor=processor,
        image_size=args.image_size,
        mode="train",
        hard_negatives_map=hard_negatives_map
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True  # Важно: отбрасываем неполный батч
    )

    # Модель
    print("Creating SigLIP Reranker model...")
    model = SigLIPReranker(
        base_model=base_model,
        projection_dim=args.projection_dim,
        freeze_base=args.freeze_base
    )
    model = model.to(device)

    # Loss и optimizer
    loss_fn = ContrastiveLoss(temperature=args.temperature)

    if args.freeze_base:
        trainable_params = [
            {'params': model.img_projection.parameters()},
            {'params': model.text_projection.parameters()}
        ]
        print("Training projection heads only (base model frozen)")
    else:
        trainable_params = model.parameters()
        print("Training full model (base + projections)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Обучение
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Temperature: {args.temperature}, Hard negatives: {args.num_hard_negatives}")

    best_recall_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*80)

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Валидация
        if epoch % args.save_every == 0 or epoch == args.epochs:
            print("\nValidating...")
            recall_top1, recall_top5 = validate(model, dataset_extract, device, args.batch_size)
            print(f"Recall@1: {recall_top1:.4f}, Recall@5: {recall_top5:.4f}")

            if recall_top1 > best_recall_top1:
                best_recall_top1 = recall_top1
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_top1': recall_top1,
                    'recall_top5': recall_top5,
                    'args': vars(args),
                    'config': {
                        'model_name': args.base_model,
                        'img_size': args.image_size,
                        'projection_dim': args.projection_dim,
                    }
                }
                torch.save(checkpoint, output_dir / "best_model.pth")
                print(f"Saved best model (recall@1: {recall_top1:.4f})")

            if epoch % args.save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall_top1': recall_top1,
                    'recall_top5': recall_top5,
                    'args': vars(args),
                    'config': {
                        'model_name': args.base_model,
                        'img_size': args.image_size,
                        'projection_dim': args.projection_dim,
                    }
                }
                torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pth")

    # Финальная модель
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'recall_top1': recall_top1,
        'recall_top5': recall_top5,
        'args': vars(args),
        'config': {
            'model_name': args.base_model,
            'img_size': args.image_size,
            'projection_dim': args.projection_dim,
        }
    }
    torch.save(final_checkpoint, output_dir / "final_model.pth")

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best Recall@1: {best_recall_top1:.4f}")
    print(f"Final Recall@1: {recall_top1:.4f}, Recall@5: {recall_top5:.4f}")
    print(f"Models saved to: {output_dir}")
    print('='*80)


if __name__ == "__main__":
    main()