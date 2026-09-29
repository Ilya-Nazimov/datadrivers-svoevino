#!/usr/bin/env python3
"""Скачивание моделей и кэшей, которые не хранятся в git.

Модели и предрассчитанные кэши занимают ~4.4 ГБ, поэтому в репозитории их нет —
они лежат в облачном хранилище. Этот скрипт восстанавливает рабочее состояние
после `git clone`.

Использование:

    python fetch_assets.py                 # скачать и распаковать
    python fetch_assets.py --check         # только проверить, что всё на месте
    python fetch_assets.py --dir /tmp/wine # сложить ассеты в другой каталог

После загрузки запустите `python prefetch_models.py` — он докачает внешние
модели (DINOv2, SigLIP, EasyOCR) из своих источников.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Источники. Подставьте свои идентификаторы Google Drive.
#  Для Google Drive file_id берётся из ссылки:
#      https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#      https://drive.google.com/uc?id=<FILE_ID>
# --------------------------------------------------------------------------- #

ARCHIVE = {
    # file_id из ссылки https://drive.google.com/file/d/<FILE_ID>/view
    "id": "1cA8a6NxBdMsByKRrRVqRXqUg5sm5st4x",
    # ожидаемый размер архива, МБ — защита от недокачанного файла
    "size_mb": 3743,
}

# Ожидаемое содержимое после распаковки: путь -> минимальный размер, МБ
# (0 = файл должен существовать, размер не проверяем)
EXPECTED = {
    "runs/wine_v2/best.pt": 330,
    "runs/siglip_enhanced_old/siglip_enhanced_last.pth": 780,
    "runs/siglip_reranker/final_model.pth": 2300,
    "yolov8x-worldv2.pt": 135,
    "weights/clip/ViT-B-32.pt": 330,
    "images/images_match.csv": 0,
    "wine.csv": 0,
    "ocr_cache/gallery_ocr_easyocr_v1.json": 0,
    # Критично: без meta.json load_sift_cache() падает при старте сервиса.
    # Файлы небольшие, но keypoint_cache/meta.json однажды потерялся из-за правила
    # `*.json` в .gitignore — поэтому проверяем их отдельно и жёстко.
    "keypoint_cache/meta.json": 0,
    "emb_cache/meta.json": 0,
}

# Каталоги, в которых должно быть хотя бы столько файлов
EXPECTED_DIRS = {
    "images": 4000,
    "keypoint_cache": 4000,
    "emb_cache": 20,
}

GDrive = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
# Почему не обычный https://drive.google.com/uc?export=download&id=... :
# для файлов больше ~100 МБ Google отдаёт HTML-страницу с подтверждением вместо
# файла (проверено: ответ 303 -> text/html). Хост usercontent.google.com с
# confirm=t отдаёт сразу application/octet-stream.


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} МБ"


def check(root: Path) -> int:
    """Проверяет наличие ассетов. Возвращает число проблем."""
    problems = 0
    for rel, min_mb in EXPECTED.items():
        p = root / rel
        if not p.exists():
            print(f"  ОТСУТСТВУЕТ  {rel}")
            problems += 1
            continue
        size_mb = p.stat().st_size / 1024 / 1024
        if min_mb and size_mb < min_mb * 0.98:
            print(f"  МАЛО        {rel}: {human(int(size_mb))} < {min_mb} МБ")
            problems += 1
        else:
            print(f"  OK           {rel}")
    for rel, min_files in EXPECTED_DIRS.items():
        d = root / rel
        n = len(list(d.iterdir())) if d.is_dir() else 0
        if n < min_files:
            print(f"  МАЛО        {rel}/: {n} файлов < {min_files}")
            problems += 1
        else:
            print(f"  OK           {rel}/: {n} файлов")
    return problems


def download_archive(url: str, dest: Path, expect_mb: int) -> None:
    print(f"Скачивание архива (~{expect_mb} МБ). Это надолго, не прерывайте.")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    seen = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            seen += len(chunk)
            if seen % (64 << 20) < (1 << 20):
                pct = seen / 1024 / 1024 / expect_mb * 100
                print(f"  {human(seen)} / ~{expect_mb} МБ ({pct:.0f}%)", flush=True)
    print(f"Скачано {human(seen)}")


def extract(archive: Path, root: Path) -> None:
    print("Распаковка...")
    # Безопасная распаковка: отсекаем абсолютные пути и выход за пределы корня
    dest = root.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest) + os.sep) and target != dest:
                raise SystemExit(f"Недопустимый путь в архиве: {member.name}")
        tf.extractall(dest)
    print("Готово.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".", help="куда класть ассеты (корень репозитория)")
    ap.add_argument("--check", action="store_true", help="только проверить")
    ap.add_argument("--url", default="", help="прямая ссылка на архив (переопределяет Google Drive)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    print(f"Каталог: {root}\n")

    if args.check:
        problems = check(root)
        if problems:
            print(f"\nПроблем: {problems}. Запустите: python fetch_assets.py")
            return 1
        print("\nВсе ассеты на месте.")
        return 0

    missing = check(root)
    if not missing:
        print("\nВсе ассеты уже на месте — скачивать не нужно.")
        return 0

    file_id = ARCHIVE["id"]
    if file_id.startswith("REPLACE"):
        print(
            "В fetch_assets.py не подставлен file_id архива.\n"
            "Укажите его в ARCHIVE['id'] или передайте прямую ссылку: --url ...",
            file=sys.stderr,
        )
        return 2
    url = args.url or (GDrive + file_id)

    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "assets.tar.gz"
        download_archive(url, archive, ARCHIVE["size_mb"])
        extract(archive, root)

    print("\nПроверяю результат:")
    if check(root):
        print("\nЧасть файлов отсутствует после распаковки — обратитесь к списку выше.")
        return 1
    print("\nДалее: python prefetch_models.py && python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
