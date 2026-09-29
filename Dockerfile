# 3.11, а не 3.12: torch публикует wheels только до cp311 (sdist отсутствует),
# поэтому на python:3.12-slim шаг pip install падает с "No matching distribution".
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip рвётся на крупных файлах: сеть обрывает загрузку (сборка падала на 51-й минуте
# с ReadTimeoutError). Увеличенные таймаут и число попыток это лечат.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

WORKDIR /app

COPY requirements.txt .
# ultralytics тянет opencv-python (GUI), easyocr — opencv-python-headless. Оба кладут модуль
# `cv2` и перетирают друг друга при установке. В контейнере без дисплея нужен только headless,
# поэтому сносим оба и ставим headless последним: так cv2 детерминированно из headless.
# cache-mount: скачанные пакеты переиспользуются между сборками и НЕ попадают в образ.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-deps opencv-python-headless==4.9.0.80

ENV HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

COPY prefetch_models.py .
# Предзагружаем ВСЕ внешние модели на этапе сборки. Иначе контейнер тянет их из сети при
# каждом старте: torch.hub тянет репозиторий dinov2 с GitHub и веса с fbaipublicfiles,
# transformers тянет siglip с HuggingFace, easyocr — модели распознавания текста.
# Без этого образ не автономен: в изолированной сети DINOv2/SigLIP/EasyOCR не загрузятся,
# а при обрыве загрузки torch.load падает с EOFError (воспроизведено на сборке).
RUN python prefetch_models.py && du -sh /root/.cache/torch /root/.cache/huggingface /root/.EasyOCR

# С этого момента сеть в рантайме не нужна: иначе библиотеки будут пытаться ходить наружу.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    YOLO_OFFLINE=True

COPY app.py .
# Обязателен: app.py:795 импортирует SigLIPReranker отсюда, чтобы загрузить обученные
# projection-головы из runs/siglip_reranker. Без него срабатывает except ImportError и
# модель молча падает на SigLIPEncoder (strict=False) — качество распознавания деградирует
# без единой ошибки в логах.
COPY siglip_negative_train.py .

# Модели (копируем только нужные)
# app.py грузит ровно 3 чекпоинта (см. app.py:78-80) — остальное из runs/ (~4.4 ГБ) не копируем
COPY runs/wine_v2/best.pt ./runs/wine_v2/best.pt
COPY runs/siglip_enhanced_old/siglip_enhanced_last.pth ./runs/siglip_enhanced_old/siglip_enhanced_last.pth
COPY runs/siglip_reranker/final_model.pth ./runs/siglip_reranker/final_model.pth
COPY yolov8x-worldv2.pt .

# Данные
COPY images/ ./images/
COPY wine.csv .

# Кэши
COPY keypoint_cache/ ./keypoint_cache/
COPY ocr_cache/ ./ocr_cache/
COPY emb_cache/ ./emb_cache/

EXPOSE 8080
ENV PORT=8080

CMD ["python", "app.py"]
