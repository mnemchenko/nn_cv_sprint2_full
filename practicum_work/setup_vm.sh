#!/usr/bin/env bash
# Установка всех зависимостей на ВМ. Запускать из корня проекта.
# Предполагаем, что torch с CUDA уже установлен (стандартный Yandex-образ).
# Если нет — раскомментировать соответствующую строку под версию CUDA.

set -euo pipefail

# --- torch (если ещё нет; раскомментируйте под свой CUDA) -----------
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# --- mmsegmentation стек (mim ставит совместимые версии) -----------
pip install -q openmim
mim install -q "mmengine>=0.10.0"
mim install -q "mmcv>=2.0.0,<2.2.0"
# наш форк mmsegmentation ставим editable
pip install -q -e .

# --- доп. пакеты под наш пайплайн -----------------------------------
pip install -q clearml opencv-python-headless scipy pycocotools

# --- ClearML credentials --------------------------------------------
# (запустить ОДИН РАЗ интерактивно перед первым обучением)
echo ""
echo "Не забудьте: clearml-init  (вставить creds c https://app.clear.ml/profile)"

# --- быстрая проверка ------------------------------------------------
python -c "import torch, mmcv, mmengine, mmseg, clearml; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('mmcv', mmcv.__version__, '| mmengine', mmengine.__version__, '| mmseg', mmseg.__version__); print('clearml ok')"
