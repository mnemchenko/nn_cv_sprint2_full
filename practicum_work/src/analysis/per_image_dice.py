"""Подсчёт per-image mDice на одном сплите для обученной модели + дамп
top-N лучших и worst-N худших предсказаний (картинка | GT | prediction).

Используется на Этапах 3 и 4 для анализа качества обучения и выбора лучшего
эксперимента; основа отчётных артефактов в `supplementary/viz/`.

Пример:
    python practicum_work/src/analysis/per_image_dice.py \\
        --config practicum_work/configs/baseline_deeplabv3plus_r50_catdog.py \\
        --checkpoint work_dirs/baseline_deeplabv3plus_r50_catdog/best_mDice_iter_*.pth \\
        --split test \\
        --out practicum_work/supplementary/viz/best_test \\
        --n 5
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmseg.registry import MODELS

PALETTE = np.array([[0, 0, 0], [220, 20, 60], [0, 200, 0]], dtype=np.uint8)
CLASS_NAMES = ("background", "cat", "dog")


def per_class_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    """Dice per class. Если класса нет ни в pred, ни в gt -> NaN (исключается из mean)."""
    out = []
    for c in range(num_classes):
        p = pred == c; g = gt == c
        s = p.sum() + g.sum()
        out.append(np.nan if s == 0 else 2.0 * (p & g).sum() / s)
    return np.array(out, dtype=np.float64)


def predict(model, dataset, idx, device):
    """Полный inference через test_step (он внутри сам зовёт data_preprocessor +
    predict + post-process — не лезем в низкоуровневое encode_decode)."""
    sample = dataset[idx]
    img = sample["inputs"].unsqueeze(0).to(device).float()
    batch = dict(inputs=img, data_samples=[sample["data_samples"]])
    with torch.no_grad():
        results = model.test_step(batch)
    return results[0].pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)


def save_triplet(img_path, gt, pred, dice_mean, out_path):
    img = np.asarray(Image.open(img_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(img); axes[0].set_title("image"); axes[0].axis("off")
    axes[1].imshow((0.55 * img + 0.45 * PALETTE[gt]).astype(np.uint8))
    axes[1].set_title("GT"); axes[1].axis("off")
    axes[2].imshow((0.55 * img + 0.45 * PALETTE[pred]).astype(np.uint8))
    axes[2].set_title(f"pred (mDice={dice_mean:.3f})"); axes[2].axis("off")
    plt.suptitle(Path(img_path).stem, fontsize=10)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--out", required=True, type=Path, help="папка под визуализации")
    ap.add_argument("--n", type=int, default=5, help="сколько best/worst отдамп")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    dl_cfg = getattr(cfg, f"{args.split}_dataloader")
    dataset = Runner.build_dataloader(dl_cfg).dataset
    print(f"split={args.split}, n={len(dataset)}")

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model = model.to(args.device).eval()

    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        s = dataset[i]
        gt = s["data_samples"].gt_sem_seg.data.cpu().numpy()[0].astype(np.uint8)
        pred = predict(model, dataset, i, args.device)
        dpc = per_class_dice(pred, gt, num_classes=len(CLASS_NAMES))
        mdice = np.nanmean(dpc)
        rows.append((s["data_samples"].img_path, mdice, dpc))

    # CSV со всеми результатами
    with open(args.out / "per_image_dice.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mDice", *[f"dice_{c}" for c in CLASS_NAMES]])
        for ip, m, dpc in rows:
            w.writerow([Path(ip).name, f"{m:.4f}", *[f"{d:.4f}" if not np.isnan(d) else "" for d in dpc]])
    mdices = np.array([r[1] for r in rows])
    print(f"mDice over split: mean={mdices.mean():.4f}  median={np.median(mdices):.4f}")

    # лучшие N и худшие N
    order = np.argsort(mdices)
    worst = order[: args.n]
    best  = order[::-1][: args.n]
    for tag, idxs in (("worst", worst), ("best", best)):
        d = args.out / tag; d.mkdir(exist_ok=True)
        for k, i in enumerate(idxs):
            ip, m, _ = rows[i]
            pred = predict(model, dataset, i, args.device)
            gt = dataset[i]["data_samples"].gt_sem_seg.data.cpu().numpy()[0].astype(np.uint8)
            save_triplet(ip, gt, pred, m, d / f"{tag}_{k:02d}_{Path(ip).stem}.png")
        print(f"saved {len(idxs)} {tag} examples -> {d}")


if __name__ == "__main__":
    main()
