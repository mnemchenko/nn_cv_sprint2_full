"""Стартовая гипотеза 2 — DeepLabV3+ с ResNet50 (Спринт 2 cat/dog).

По учебнику: DeepLab выбираем, когда есть классы из ImageNet/COCO. У нас
ровно эти классы (cat, dog), поэтому ожидаем выигрыш относительно UNet за счёт
богатого ImageNet pretrain'а ResNet50_v1c (open-mmlab://resnet50_v1c).
"""
_base_ = [
    "../_base_/models/deeplabv3plus_r50-d8.py",
    "../_base_/datasets/practice_dataset.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/practice_schedule.py",
]

crop_size = (256, 256)
num_classes = 3

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

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(
            type="ClearMLVisBackend",
            init_kwargs=dict(
                project_name="YaPracticum",
                task_name="deeplabv3plus_r50-d8_1xb16-practice_dataset-256x256",
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
