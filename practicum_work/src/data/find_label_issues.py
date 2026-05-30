"""Поиск кандидатов на ошибки разметки в формате mmsegmentation.

Идея: у корректной маски граница идёт по реальным краям объекта, поэтому средний
градиент изображения на границе маски заметно выше, чем по картинке в среднем
(edge-alignment > 1). Если маска лежит на людях/мебели/гладком фоне, её граница
краёв не касается -> низкий edge-alignment. Плюс отдельно отмечаются вырожденные
маски (почти пустые).

Это НЕ ловит перепутанный класс cat<->dog (маска на животном, просто не того класса) —
такие смотрятся глазами по монтажам montage_cat/montage_dog из eda.ipynb.

Пример:
    python find_label_issues.py --root ../../../data/segmentation_dataset \
        --split train --topk 25
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

DEGENERATE_AREA = 5    # маска меньше — считаем вырожденной


def sobel_mag(gray):
    return np.hypot(ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0))


def score_split(root: Path, split: str):
    img_dir = root / "img" / split
    lbl_dir = root / "labels" / split
    rows = []
    for ip in sorted(img_dir.glob("*.jpg")):
        lp = lbl_dir / f"{ip.stem}.png"
        mask = np.asarray(Image.open(lp))
        fg = mask > 0
        area = int(fg.sum())
        cls = "cat" if 1 in np.unique(mask) else ("dog" if 2 in np.unique(mask) else "bg")
        if area < DEGENERATE_AREA:
            rows.append((ip.stem, 0.0, area / mask.size, cls, "degenerate"))
            continue
        boundary = ndimage.binary_dilation(fg & ~ndimage.binary_erosion(fg), iterations=1)
        g = sobel_mag(np.asarray(Image.open(ip).convert("L"), dtype=np.float64))
        align = float(g[boundary].mean() / (g.mean() + 1e-6))
        rows.append((ip.stem, align, area / mask.size, cls, ""))
    rows.sort(key=lambda r: r[1])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--topk", type=int, default=25, help="сколько худших показать")
    args = ap.parse_args()

    rows = score_split(args.root, args.split)
    aligns = np.array([r[1] for r in rows])
    print(f"[{args.split}] n={len(rows)}  "
          f"edge-align median={np.median(aligns):.2f} p10={np.percentile(aligns, 10):.2f}")
    print(f"\nworst {args.topk} (низкий align = маска вероятно не на объекте):")
    print(f'{"file":<22}{"align":>7}{"fg%":>8}  class  flag')
    for stem, al, fr, cl, flag in rows[:args.topk]:
        print(f"{stem:<22}{al:>7.2f}{fr * 100:>7.1f}%  {cl:<5}  {flag}")


if __name__ == "__main__":
    main()
