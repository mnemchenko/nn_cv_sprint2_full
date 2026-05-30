"""Конвертация датасета из формата mmsegmentation (grayscale-PNG маски) в COCO
для загрузки в CVAT и ручной доразметки.

Формат входа (mmsegmentation):
    <root>/img/<split>/<name>.jpg
    <root>/labels/<split>/<name>.png      # 8-bit grayscale, значения = id класса

Формат выхода (COCO instances):
    <out>/images/<name>.jpg               # копии выбранных изображений
    <out>/annotations.json                # COCO с полигонами текущих масок

Каждая связная компонента каждого foreground-класса экспортируется как отдельный
полигон-инстанс — так маску удобно править в CVAT. Полигоны строятся из контуров
текущих (грубых) масок, чтобы аннотатор начинал не с нуля, а правил готовое.

Примеры:
    # экспортировать только подозрительные семплы (по списку файлов)
    python mmsegmentation_to_coco.py --root ../../../data/segmentation_dataset \
        --split train --out ../../../data/cvat_export \
        --files 000000028253_7169 000000049758_3963 000000481101_7500

    # экспортировать весь split
    python mmsegmentation_to_coco.py --root ../../../data/segmentation_dataset \
        --split train --out ../../../data/cvat_export_train
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# id класса в маске -> имя категории COCO. background (0) категорией не является.
CLASSES = {1: "cat", 2: "dog"}
MIN_AREA = 10          # отбрасываем мусорные компоненты меньше этого числа пикселей
MIN_POLY_POINTS = 3    # COCO-полигон должен иметь >= 3 вершин


def mask_to_polygons(binary_mask):
    """Контуры одной бинарной маски -> список COCO-полигонов [[x,y,x,y,...], ...]."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        seg = cnt.reshape(-1).astype(float).tolist()
        if len(seg) >= MIN_POLY_POINTS * 2:
            polygons.append(seg)
    return polygons


def build_coco(root: Path, split: str, files):
    img_dir = root / "img" / split
    lbl_dir = root / "labels" / split

    if files:
        stems = list(files)
    else:
        stems = sorted(p.stem for p in img_dir.glob("*.jpg"))

    images, annotations = [], []
    ann_id = 1
    for img_id, stem in enumerate(stems, start=1):
        ip = img_dir / f"{stem}.jpg"
        lp = lbl_dir / f"{stem}.png"
        if not ip.exists() or not lp.exists():
            print(f"  ! пропуск (нет файла): {stem}")
            continue
        mask = np.asarray(Image.open(lp))
        h, w = mask.shape[:2]
        images.append({"id": img_id, "file_name": f"{stem}.jpg", "width": int(w), "height": int(h)})

        for cls_id, cls_name in CLASSES.items():
            for poly in mask_to_polygons(mask == cls_id):
                xs, ys = poly[0::2], poly[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "segmentation": [poly],
                    "area": float(cv2.contourArea(np.array(poly).reshape(-1, 1, 2).astype(np.float32))),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "iscrowd": 0,
                })
                ann_id += 1

    categories = [{"id": cid, "name": name, "supercategory": ""} for cid, name in CLASSES.items()]
    coco = {
        "info": {"description": f"cat/dog segmentation, split={split}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, [img["file_name"] for img in images]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="корень датасета mmsegmentation")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, type=Path, help="выходная папка COCO")
    ap.add_argument("--files", nargs="*", default=None,
                    help="список stem'ов файлов (без расширения); если не задан — весь split")
    args = ap.parse_args()

    coco, file_names = build_coco(args.root, args.split, args.files)

    out_img = args.out / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    for fn in file_names:
        shutil.copy(args.root / "img" / args.split / fn, out_img / fn)
    with open(args.out / "annotations.json", "w") as f:
        json.dump(coco, f)

    print(f"exported: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"  images      -> {out_img}")
    print(f"  annotations -> {args.out / 'annotations.json'}")
    print("В CVAT: создать task, импортировать аннотации в формате 'COCO 1.0'.")


if __name__ == "__main__":
    main()
