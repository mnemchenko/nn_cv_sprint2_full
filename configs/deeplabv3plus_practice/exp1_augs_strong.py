"""Эксперимент 1 — усиленные аугментации.

Меняем ровно ОДНУ вещь относительно бейзлайна — `train_pipeline`:
  * `PhotoMetricDistortion` с дефолтными диапазонами mmseg (вместо лёгких);
  * добавлен `RandomRotFlip` (90° повороты + flip из урока 7);
  * добавлен `RandomCutOut` — регуляризация на малом датасете (179 семплов).

Всё остальное (модель, лосс CE+Dice 1:1, schedule, optimizer, batch) — наследуется
от бейзлайна. Мотивация — на 179 семплах баланс bias↔variance смещён к
переобучению, агрессивные аугментации должны помочь.
"""
_base_ = "./deeplabv3plus_r50-d8_1xb16-practice_dataset-256x256.py"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PhotoMetricDistortion"),   # дефолтные диапазоны (сильнее baseline)
    dict(type="RandomRotFlip"),           # из урока 7
    dict(type="RandomCutOut", prob=0.5,
         n_holes=(3, 8), cutout_ratio=(0.05, 0.12)),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(
            type="ClearMLVisBackend",
            init_kwargs=dict(
                project_name="YaPracticum",
                task_name="exp1_augs_strong",
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
