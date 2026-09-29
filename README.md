# Wine Recognition Service

Сервис распознавания винных этикеток: принимает фотографию, возвращает Top-1 `slug` вина.

В репозитории — **только код и данные**. Модели и предрассчитанные кэши (~4.4 ГБ)
лежат в облачном хранилище и скачиваются скриптом `fetch_assets.py`. Хранить такие
бинарники в git непрактично: они не диффятся, каждый клон тянет гигабайты, а история
раздувается навсегда.

---

## Требования

| Компонент | Версия | Почему важно |
|---|---|---|
| **Python** | **3.11** | **не 3.12.** `torch 2.5.1` публикует wheels только до `cp311`, sdist отсутствует. На 3.12 установка падает с `No matching distribution` |
| Сеть | да | ~4 ГБ пакетов + ~4.4 ГБ ассетов + ~800 МБ внешних моделей |
| ОС | Linux / macOS | протестировано на macOS (Apple Silicon) и Linux ARM64 |
| GPU | не требуется | считает на CPU |

---

## Установка

```bash
git clone <url-repo> && cd <repo>

# 1. Зависимости
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Модели и кэши (~4.4 ГБ) из облачного хранилища
python fetch_assets.py
python fetch_assets.py --check     # убедиться, что всё на месте

# 3. Внешние модели: DINOv2, SigLIP, EasyOCR, CLIP (~1.2 ГБ)
python prefetch_models.py

# 4. Запуск (из корня репозитория — все пути относительные)
python app.py
```

Сервис поднимется на `http://127.0.0.1:8080`. Первый запуск занимает **1–3 минуты**.

Проверка готовности:

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok","gallery_size":4218}
```

---

## Прогон валидации

```bash
cd eval
./participant_test.sh \
  --images-dir ./queries \
  --manifest ./queries.tsv \
  --endpoint 'http://127.0.0.1:8080/v1/eval/predict' \
  --output ./predictions.jsonl
```

Требования на стороне машины, где запускается скрипт: `bash`, `curl`, `jq`, `awk`.
Подробности — в [`eval/README.md`](eval/README.md).

Свой набор фото: положите изображения в папку и перечислите их в манифесте
`queries.tsv` (формат `query_id<TAB>image_path`, обязателен заголовок
`query_id<TAB>image_path`).

---

## Структура

```
app.py                     основной сервис (FastAPI)
siglip_negative_train.py   нужен app.py:795 — без него SigLIP молча деградирует
build_caches.py            пересчёт кэшей галереи
prefetch_models.py         предзагрузка внешних моделей (DINOv2, SigLIP, EasyOCR, CLIP)
fetch_assets.py            скачивание моделей и кэшей из облачного хранилища

# --- в репозитории (git) ---
wine.csv                   соответствие бутылка → вино
eval/                      скрипт валидации организатора

# --- скачиваются из облачного хранилища (fetch_assets.py) ---
runs/wine_v2/best.pt                          DINOv2 + ArcFace
runs/siglip_enhanced_old/…_last.pth            SigLIP image-эмбеддинги
runs/siglip_reranker/final_model.pth          SigLIP text-reranker
yolov8x-worldv2.pt                            детекция бутылки
weights/clip/ViT-B-32.pt                     CLIP для YOLO-World
images/                    галерея (4218 фото) + images_match.csv
keypoint_cache/            SIFT-дескрипторы галереи (по .npz на фото)
emb_cache/                 эмбеддинги галереи (DINOv2 и SigLIP)
ocr_cache/                 распознанный текст этикеток
```

### Состав архива `wine_assets.tar.gz`

Архив с моделями, галереей и кэшами (**3.65 ГБ**):

```
https://drive.google.com/file/d/1cA8a6NxBdMsByKRrRVqRXqUg5sm5st4x/view
```

Контрольная сумма (совпадёт, если файл скачался целиком):

```
sha256  ee4ef06670e373f89f147417f34fd3574eb8588e770d2bf9171560f78c9c675b
размер  3 924 423 279 байт
```

```bash
shasum -a 256 wine_assets.tar.gz
```

Пути внутри архива совпадают с путями в корне репозитория — распаковывать нужно
ровно туда, без пересоздания структуры. После распаковки проверьте состав:

```bash
python fetch_assets.py --check
```

### Про кэши

`keypoint_cache/`, `emb_cache/` и `ocr_cache/` — **не опциональны**, а обязательны для
работы. Без них сервис пересчитывает SIFT и эмбеддинги для 4218 изображений при
каждом старте: десятки минут вместо секунд. Поэтому они лежат в облаке вместе с
моделями, а не генерируются на месте.

Пересобрать: `python build_caches.py`.

---

## Производительность

Инференс идёт на CPU. Крупные фото автоматически ужимаются до 1500 px по высоте
(`MAX_QUERY_HEIGHT` в `app.py`) — качества хватает для сопоставления по SIFT/SigLIP,
а нагрузка на CPU падает кратно.

Лимит клиента — 15 секунд (`--max-time` в `eval/participant_test.sh`).

**На Apple Silicon запускайте с `WINE_FORCE_CPU=1`:** DINOv2 вызывает
`upsample_bicubic2d`, которого на MPS нет, и с MPS-fallback пересчёт галереи
занимает непозволительно долго.

---

## Известные особенности сборки

Набор пакетов зафиксирован полностью (включая транзитивные зависимости) — так
воспроизводимость не зависит от изменений в PyPI.

Три неочевидных момента, на которые стоит обратить внимание при обновлении:

1. **opencv.** `ultralytics` требует `opencv-python`, `easyocr` — `opencv-python-headless`.
   Оба содержат модуль `cv2` и перетирают друг друга при установке. В `Dockerfile` оба
   сносятся и ставится только headless (контейнеру без дисплея GUI-версия не нужна).
2. **transformers.** Ниже 4.38 нет `SiglipProcessor`/`SiglipModel` — `AutoProcessor`
   падает с `Unrecognized processing class`, и три стадии инференса из пяти не работают.
3. **torch.** Ниже 2.2 на Linux ARM64 падает с Segmentation fault базовый `nn.LSTM`
   (oneDNN-rnn), из-за чего easyocr падает на первом же `readtext`. Баг не воспроизводится
   на macOS/x86.

CUDA-зависимости (`nvidia-*`, `triton`) намеренно не перечислены в `requirements.txt`:
`torch` тянет их сам по маркеру `platform_system == "Linux"`. На машине без GPU они
бесполезны, но `torch 2.14.0` падает на `import torch` без них (`libcublasLt.so not found`),
так что исключать их через `--no-deps` нельзя.

---

## Docker (необязательно)

```bash
docker build -t wine-recognizer .
docker run --rm -p 8080:8080 wine-recognizer
```

Образ содержит все модели, включая внешние, и работает без сети. Собран под
`linux/arm64`; на других платформах нужен `docker buildx build --platform`.
