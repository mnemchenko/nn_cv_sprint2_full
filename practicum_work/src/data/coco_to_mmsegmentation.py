"""Конвертация разметки из COCO (экспорт из CVAT после доразметки) обратно в формат
mmsegmentation (8-bit grayscale-PNG, значение пикселя = id класса).

Категории сопоставляются по ИМЕНИ (а не по id), потому что CVAT может перенумеровать
категории при экспорте:
    background -> 0   (всё, что не размечено)
    cat        -> 1
    dog        -> 2

Поддерживаются оба вида segmentation в COCO: полигоны (list) и RLE (dict).
Если на картинку пришло несколько аннотаций — они «запекаются» по очереди, более
поздняя перекрывает более раннюю (для cat/dog-датасета на картинке один класс,
конфликтов нет).

Пример:
    python coco_to_mmsegmentation.py \
        --coco ../../../data/cvat_fixed/annotations.json \
        --out-labels ../../../data/segmentation_dataset/labels/train
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

NAME_TO_ID = {"background": 0, "cat": 1, "dog": 2}


def render_mask(anns, h, w, catid_to_name):
    """Список аннотаций одной картинки -> grayscale-маска (uint8) с id классов."""
    out = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        name = catid_to_name.get(ann["category_id"])
        cls = NAME_TO_ID.get(name)
        if not cls:  # None или 0 (background) — пропускаем
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, list):  # полигоны
            for poly in seg:
                pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(out, [pts], int(cls))
        elif isinstance(seg, dict):  # RLE (SAM2 / mask-фигуры из CVAT)
            rle = dict(seg)
            counts = rle.get("counts")
            if isinstance(counts, list):       # uncompressed RLE -> compressed
                rle = mask_utils.frPyObjects(rle, h, w)
            elif isinstance(counts, str):      # CVAT кладёт counts строкой, pycocotools ждёт bytes
                rle = {"size": rle["size"], "counts": counts.encode("utf-8")}
            m = mask_utils.decode(rle)
            out[m > 0] = cls
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, type=Path, help="COCO json из CVAT")
    ap.add_argument("--out-labels", required=True, type=Path, help="куда писать PNG-маски")
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text())
    catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    args.out_labels.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in coco["images"]:
        h, w = img["height"], img["width"]
        mask = render_mask(anns_by_img.get(img["id"], []), h, w, catid_to_name)
        stem = Path(img["file_name"]).stem
        Image.fromarray(mask, mode="L").save(args.out_labels / f"{stem}.png")
        n += 1

    print(f"wrote {n} masks -> {args.out_labels}")
    print("Эти PNG можно класть прямо в labels/<split>, заменяя исходные.")


if __name__ == "__main__":
    main()
