#!/usr/bin/env bash
# Установка зависимостей на ВМ Яндекс Практикума (T4 / cu118).
# СТРОГО по уроку Спринта 2 "Получение ВМ" — версии torch/mmcv/numpy и т.д.
# зафиксированы потому, что:
#   * есть предсобранная сборка mmcv 2.1.0 под torch 2.0.0 + cu118;
#   * mmcv >=2.2.0 не поддерживается mmsegmentation;
#   * numpy 2.x ломает совместимость с mmseg.
#
# Перед запуском:
#   1. Создайте и активируйте venv:
#        python3.10 -m venv ~/practicum_venv
#        source ~/practicum_venv/bin/activate
#   2. cd в корень репо (там, где setup.py)
#   3. bash practicum_work/setup_vm.sh

set -euo pipefail
PY=python3.10

echo "==> [1/6] обновление pip / setuptools / wheel"
$PY -m pip install --upgrade pip setuptools wheel

echo "==> [2/6] PyTorch 2.0.0 + cu118 (точные версии по уроку)"
$PY -m pip install \
    torch==2.0.0+cu118 \
    torchvision==0.15.1+cu118 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

echo "==> [3/6] openmim + mmengine + mmcv==2.1.0 (предсобранная под torch2.0.0/cu118)"
$PY -m pip install -U openmim
$PY -m mim install mmengine
$PY -m mim install "mmcv==2.1.0"

echo "==> [4/6] mmsegmentation (наш форк, editable)"
$PY -m pip install -v -e .

echo "==> [5/6] зафиксированные доп. зависимости из урока"
$PY -m pip install \
    ftfy==6.3.1 \
    regex==2025.9.18 \
    numpy==1.26.4 \
    clearml==2.0.2

echo "==> [6/6] доп. пакеты под наш проект (CVAT-конвертация, EDA-скрипты)"
$PY -m pip install opencv-python-headless scipy pycocotools

echo ""
echo "==> verify: версии и доступность GPU"
$PY -c "
import torch, mmcv, mmengine, mmseg, numpy, clearml
print('torch    ', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('mmcv     ', mmcv.__version__)
print('mmengine ', mmengine.__version__)
print('mmseg    ', mmseg.__version__)
print('numpy    ', numpy.__version__)
print('clearml  ', clearml.__version__)
"

echo ""
echo "============================================================"
echo "  ДАЛЬШЕ:"
echo "    1) clearml-init     (вставить creds из локального keys.txt)"
echo "    2) (опц.) проверка инсталляции demo-инференсом mmseg —"
echo "       см. урок 'Получение ВМ'."
echo "    3) python practicum_work/sanity_check.py \\"
echo "         configs/deeplabv3plus_practice/deeplabv3plus_r50-d8_1xb16-practice_dataset-256x256.py"
echo "============================================================"
