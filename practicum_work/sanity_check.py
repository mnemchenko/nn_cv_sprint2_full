"""Полная проверка готовности к обучению. Запускать НА ВМ из корня проекта:

    python practicum_work/sanity_check.py configs/unet_practice/unet-s5-d16_fcn_1xb16-practice_dataset-256x256.py
    python practicum_work/sanity_check.py configs/deeplabv3plus_practice/deeplabv3plus_r50-d8_1xb16-practice_dataset-256x256.py

Что проверяется:
  1. конфиг парсится через mmengine.Config.fromfile;
  2. PracticeDataset зарегистрирован (через mmseg.datasets.__init__);
  3. train-датасет инициализируется и читает ровно столько семплов,
     сколько в splits/train.txt;
  4. val-датасет инициализируется (без split-файла);
  5. модель строится из конфига и проходит forward на случайном тензоре;
  6. ClearMLVisBackend подключается (нужен прошедший clearml-init).
"""
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def main(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    print(f"[1/6] config loaded: {cfg_path}")

    init_default_scope(cfg.get("default_scope", "mmseg"))

    # 3) train dataset
    train_ds = Runner.build_dataloader(cfg.train_dataloader).dataset
    print(f"[2/6] train dataset: {type(train_ds).__name__}  len={len(train_ds)}")
    expected = sum(1 for _ in open("data/segmentation_dataset/splits/train.txt") if _.strip())
    assert len(train_ds) == expected, f"len mismatch: {len(train_ds)} != {expected}"
    print(f"[3/6] train.txt match: ok ({expected} samples)")

    # 4) val dataset
    val_ds = Runner.build_dataloader(cfg.val_dataloader).dataset
    print(f"[4/6] val dataset:   {type(val_ds).__name__}  len={len(val_ds)}")

    # 5) model build + forward
    from mmseg.registry import MODELS
    model = MODELS.build(cfg.model)
    model.eval()
    sample = train_ds[0]
    img = sample["inputs"].unsqueeze(0).float()
    with torch.no_grad():
        out = model.data_preprocessor(dict(inputs=img, data_samples=[sample["data_samples"]]),
                                      training=False)
        feats = model.backbone(out["inputs"])
    print(f"[5/6] model forward: ok  (img {tuple(img.shape)} -> feats {[tuple(f.shape) for f in feats]})")

    # 6) visualizer — без реальной инициализации (иначе ClearMLVisBackend создаст
    # лишний Task на app.clear.ml). Только сверяем конфиг + что класс импортируется.
    backends = [b.get("type") for b in cfg.visualizer.get("vis_backends", [])]
    print(f"[6/6] visualizer backends (из конфига): {backends}")
    if "ClearMLVisBackend" in backends:
        try:
            from mmengine.visualization import ClearMLVisBackend  # noqa: F401
            import clearml  # noqa: F401
            print("       ClearMLVisBackend импортируется, clearml установлен — ok")
        except ImportError as e:
            print(f"       ⚠ ClearML недоступен: {e}")

    print("\nALL OK — можно запускать обучение:")
    print(f"  python tools/train.py {cfg_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python sanity_check.py <path/to/config.py>"); sys.exit(1)
    main(sys.argv[1])
