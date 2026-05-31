"""Эксперимент 3 — более тяжёлый backbone ResNet101 + d16 + multi-grid.

Меняем ровно ОДНУ вещь относительно бейзлайна — **архитектуру backbone**:
  бейзлайн: ResNet50,  output stride 8 (d8),  стандартные dilations.
  здесь:    ResNet101, output stride 16 (d16), multi-grid (1,2,4) в последнем блоке;
            decoder ASPP с уменьшенными dilations (1,6,12,18) — стандарт под d16.
  pretrained: 'open-mmlab://resnet101_v1c'.

Обоснование по уроку «Современные модификации в mmseg»:
  * R101 даёт больше pretrained-капасити (cat/dog ∈ ImageNet → быстрее сойдётся);
  * multi-grid в последнем ResLayer улучшает глобальный контекст без увеличения
    числа параметров;
  * по таблице замеров скорости из урока r101-d16-mg124 (0.040 c/итер) даже
    БЫСТРЕЕ r50-d8 (0.077 c/итер), поэтому это не только «жирнее», но и эффективнее
    по скорости инференса.

Всё остальное (датасет, лосс CE+Dice 1:1, augs, schedule 100 эпох, batch=16) —
наследуется от бейзлайна.
"""
_base_ = "./deeplabv3plus_r50-d8_1xb16-practice_dataset-256x256.py"

model = dict(
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(
        depth=101,
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        multi_grid=(1, 2, 4),
    ),
    decode_head=dict(
        dilations=(1, 6, 12, 18),   # ASPP под d16
    ),
)

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(
            type="ClearMLVisBackend",
            init_kwargs=dict(
                project_name="YaPracticum",
                task_name="exp3_r101_d16_mg124",
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
