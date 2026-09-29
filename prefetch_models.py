"""Предзагрузка внешних моделей на этапе сборки Docker-образа.

Зачем: app.py в рантайме тянет DINOv2 (torch.hub, GitHub + fbaipublicfiles), SigLIP
(transformers, HuggingFace) и EasyOCR (jaided.ai). Без этого образ не автономен, а
при обрыве загрузки torch.load падает с EOFError — это воспроизводилось на сборке.

Сеть в этой среде регулярно рвёт крупные файлы, поэтому на каждую модель делаем
несколько попыток и ПЕРЕД повтором удаляем её кэш: иначе следующая попытка повторно
скачает тот же битый файл и упадёт с тем же EOFError.
"""

import os
import shutil
import sys
import time

MAX_ATTEMPTS = 6
BACKOFF_SECONDS = 10


def _dinov2_cache() -> str:
    return os.path.join(os.environ.get("TORCH_HOME", ""), "hub")


def _download_dinov2() -> None:
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)


def _download_siglip() -> None:
    from transformers import AutoModel, AutoProcessor

    AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    AutoModel.from_pretrained("google/siglip-base-patch16-224")


def _download_easyocr() -> None:
    import easyocr

    easyocr.Reader(["ru", "en"], gpu=False, verbose=False)


def _download_clip() -> None:
    # YOLO-World (yolov8x-worldv2.pt) вызывает set_classes(), а тот импортирует clip и
    # зовёт clip.load("ViT-B/32") — без этого YOLO отключается ТИХО, сервис работает,
    # но без детекции бутылки. clip грузит модель из сети, поэтому кэшируем и её.
    import clip

    clip.load("ViT-B/32")


MODELS = [
    ("dinov2 (torch.hub, ~330MB)", _dinov2_cache, _download_dinov2),
    ("siglip-base-patch16-224 (HF, ~375MB)", lambda: os.environ.get("HF_HOME"), _download_siglip),
    ("easyocr ru+en (~100MB)", lambda: os.path.expanduser("~/.EasyOCR"), _download_easyocr),
    ("clip ViT-B/32 (~340MB)", lambda: os.path.expanduser("~/.cache/clip"), _download_clip),
]


def prefetch(name: str, cache_path, download) -> None:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            download()
            print(f"[OK] {name}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 — ловим любую ошибку загрузки
            last_error = exc
            path = cache_path()
            # Чистим кэш только перед повтором: битый файл иначе будет переиспользован.
            if attempt < MAX_ATTEMPTS and path:
                shutil.rmtree(path, ignore_errors=True)
            print(
                f"[WARN] {name}: попытка {attempt}/{MAX_ATTEMPTS} не удалась: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_SECONDS * attempt)
    raise SystemExit(f"Не удалось загрузить {name}: {last_error}")


def main() -> None:
    for name, cache_path, download in MODELS:
        prefetch(name, cache_path, download)
    print("prefetch: все модели загружены", flush=True)


def _note() -> None:
    print(
        "\nПримечание про CLIP: модель кэшируется в ~/.cache/clip.\n"
        "Если она лежит в репозитории (weights/clip/ViT-B-32.pt), положите её туда\n"
        "заранее, чтобы не тянуть из сети.",
        flush=True,
    )


if __name__ == "__main__":
    rc = main()
    _note()
    sys.exit(rc)
