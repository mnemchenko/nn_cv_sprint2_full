"""Стартовая гипотеза 1 — UNet (Спринт 2 cat/dog).

По учебнику: UNet — модель без сильной зависимости от ImageNet pretrain,
используется как контрольная точка относительно DeepLabV3+ R50 (см. README).
Базовый mmseg-конфиг fcn_unet_s5-d16 с тонкой настройкой под наши данные:
  * num_classes=3 (background/cat/dog);
  * лосс CE+Dice (1.0+1.0) против перекоса в 90% фон;
  * нормализация по нашим mean/std из EDA;
  * BN вместо SyncBN (одиночный GPU);
  * test_cfg=whole (256×256 целиком, без слайдинга).
"""
_base_ = [
    "../_base_/models/fcn_unet_s5-d16.py",
    "../_base_/datasets/practice_dataset.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/practice_schedule.py",
]

crop_size = (256, 256)
num_classes = 3

# data_preprocessor с нашими mean/std (посчитаны в EDA на очищенном train) + размером
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[118.095, 111.101, 98.652],
    std=[67.270, 67.047, 68.951],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce",   loss_weight=1.0),
            dict(type="DiceLoss",         loss_name="loss_dice", loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
    test_cfg=dict(mode="whole"),
)

# Логирование: Local + ClearML (по примеру урока, проект YaPracticum)
visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(
            type="ClearMLVisBackend",
            init_kwargs=dict(
                project_name="YaPracticum",
                task_name="unet-s5-d16_fcn_1xb16-practice_dataset-256x256",
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            ),
        ),
    ],
)
