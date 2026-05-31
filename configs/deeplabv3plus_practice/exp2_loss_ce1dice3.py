"""Эксперимент 2 — баланс лосса CE:Dice = 1:3.

Меняем ровно ОДНУ вещь относительно бейзлайна — веса лосса в `decode_head`:
  бейзлайн: CE 1.0 + Dice 1.0
  здесь:    CE 1.0 + Dice 3.0

Обоснование. Готовые mmseg-конфиги для несбалансированных датасетов (например,
`unet/.../chase_db1` с дисбалансом ~10:1) используют CE:Dice = 1:3, чтобы
сильнее давить Dice. У нас 90:10 фон:foreground — дисбаланс ещё более сильный,
поэтому увеличить вес Dice по аналогии — логичная гипотеза.

Всё остальное (модель, schedule, optimizer, augs) — наследуется от бейзлайна.
"""
_base_ = "./deeplabv3plus_r50-d8_1xb16-practice_dataset-256x256.py"

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce",   loss_weight=1.0),
            dict(type="DiceLoss",         loss_name="loss_dice", loss_weight=3.0),
        ],
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
                task_name="exp2_loss_ce1_dice3",
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
