"""Базовый dataset-конфиг для PracticeDataset (cat/dog, 256×256, 3 класса).

Train читает только очищенные стемы из splits/train.txt (179 семплов после
доразметки в CVAT в рамках Этапа 1). Val/test — все файлы из соответствующих
директорий.

Пайплайн baseline-уровня по уроку: для пикчей уже 256×256 → resize не нужен,
только RandomFlip + лёгкий PhotoMetric (усиление аугментаций — в Этапе 3).
"""
dataset_type = "PracticeDataset"
data_root = "data/segmentation_dataset"
crop_size = (256, 256)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion",
         brightness_delta=16, contrast_range=(0.9, 1.1),
         saturation_range=(0.9, 1.1), hue_delta=10),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
test_pipeline = val_pipeline

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="splits/train.txt",
    data_prefix=dict(img_path="img/train", seg_map_path="labels/train"),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png",
)
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path="img/val", seg_map_path="labels/val"),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png",
)
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path="img/test", seg_map_path="labels/test"),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png",
)

train_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset,
)

# Целевая метрика — mDice (через IoUMetric).
val_evaluator = dict(type="IoUMetric", iou_metrics=["mDice", "mIoU"])
test_evaluator = val_evaluator
