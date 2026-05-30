"""Конвертация разметки из CVAT 1.1 «for images» (XML) в формат mmsegmentation.

Альтернатива coco_to_mmsegmentation.py — на случай, если из CVAT выгружают XML, а
не COCO JSON. Семантика та же: маски PNG (uint8, значение = id класса).

Поддерживаются:
  * <polygon ... points="x,y;x,y;..."/>
  * <mask ... rle="N,N,N,..." left top width height/>   (например, от SAM2)
  * метки: background=0, cat=1, dog=2 (мапинг по имени)

Изображения, в которых аннотатор удалил ВСЕ фигуры, считаются «исключёнными» из
обучения: для них не пишется маска и они попадают в drop-список. Дополнительно
пишется split-файл (train.txt / val.txt) со списком оставленных stems — этот файл
читает mmseg-датасет, чтобы загружать только их.

Пример:
    python cvat_to_mmsegmentation.py \\
        --xml ../../../data/cvat_result/annotations.xml \\
        --out-labels ../../../data/segmentation_dataset/labels/train \\
        --split-list ../../../data/segmentation_dataset/splits/train.txt
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

NAME_TO_ID = {"background": 0, "cat": 1, "dog": 2}


def parse_points(s):
    pts = [tok.split(",") for tok in s.split(";") if tok.strip()]
    return np.array([(float(x), float(y)) for x, y in pts]).round().astype(np.int32)


def decode_cvat_rle(rle_str, left, top, mw, mh, full_h, full_w):
    """CVAT 1.1 RLE для <mask>: подряд идущие длины runов, начиная с фона (0),
    чередуя 0/1, поток читается построчно внутри bbox (left, top, width=mw, height=mh)."""
    runs = [int(x) for x in rle_str.split(",")]
    flat = np.zeros(mw * mh, dtype=np.uint8)
    pos, val = 0, 0
    for run in runs:
        if val:
            flat[pos:pos + run] = 1
        pos += run
        val ^= 1
    sub = flat[:mw * mh].reshape(mh, mw)
    full = np.zeros((full_h, full_w), dtype=np.uint8)
    full[top:top + mh, left:left + mw] = sub
    return full


def _paint(shape, mask, cls):
    """Закрасить фигуру (polygon | mask) указанным классом. Возвращает True, если что-то нарисовано."""
    h, w = mask.shape
    if shape.tag == "polygon":
        pts = parse_points(shape.attrib["points"])
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(cls))
            return True
    elif shape.tag == "mask":
        left = int(round(float(shape.attrib["left"])))
        top = int(round(float(shape.attrib["top"])))
        mw = int(round(float(shape.attrib["width"])))
        mh = int(round(float(shape.attrib["height"])))
        m = decode_cvat_rle(shape.attrib["rle"], left, top, mw, mh, h, w)
        if m.any():
            mask[m > 0] = cls
            return True
    return False


def render_image(img_elem, h, w):
    """Прорисовать все фигуры одной картинки -> (mask, has_any_shape).

    Двухпроходная отрисовка, чтобы порядок шейпов в XML не ломал семантику:
      1) сначала ВСЕ foreground-шейпы (cat / dog);
      2) затем ВСЕ background-шейпы — они «вычитают» дырки в животном
         (полигоны background используются аннотатором, чтобы прорезать проёмы).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    has_fg = False
    # pass 1: foreground (cat / dog)
    for shape in img_elem:
        label = shape.attrib.get("label")
        if NAME_TO_ID.get(label, 0) > 0:
            has_fg |= _paint(shape, mask, NAME_TO_ID[label])
    # pass 2: background (вычитание дырок) — рисуется только если есть foreground;
    # картинка с одними background-полигонами считается «без аннотации» (drop).
    if has_fg:
        for shape in img_elem:
            if shape.attrib.get("label") == "background":
                _paint(shape, mask, 0)
    return mask, has_fg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, type=Path)
    ap.add_argument("--out-labels", required=True, type=Path)
    ap.add_argument("--split-list", type=Path, default=None,
                    help="куда записать stems с оставленными изображениями (по строке)")
    ap.add_argument("--print-dropped", action="store_true",
                    help="распечатать список исключённых файлов")
    args = ap.parse_args()

    root = ET.parse(args.xml).getroot()
    args.out_labels.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], []
    for img in root.findall("image"):
        name = img.attrib["name"]
        h = int(img.attrib["height"]); w = int(img.attrib["width"])
        mask, has_any = render_image(img, h, w)
        stem = Path(name).stem
        if has_any:
            Image.fromarray(mask, mode="L").save(args.out_labels / f"{stem}.png")
            kept.append(stem)
        else:
            dropped.append(stem)

    print(f"kept   : {len(kept):>3}  -> masks written to {args.out_labels}")
    print(f"dropped: {len(dropped):>3}  (no annotations -> excluded from split file)")
    if args.print_dropped and dropped:
        print("dropped stems:")
        for s in dropped:
            print(f"  {s}")
    if args.split_list:
        args.split_list.parent.mkdir(parents=True, exist_ok=True)
        args.split_list.write_text("\n".join(kept) + "\n")
        print(f"split  -> {args.split_list}")


if __name__ == "__main__":
    main()
