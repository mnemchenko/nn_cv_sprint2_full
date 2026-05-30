"""Гиперпараметры обучения для cat/dog проекта (Спринт 2).

EpochBasedTrainLoop по уроку. С batch=16 и 179 train семплов — ~12 итер/эпоху,
100 эпох ≈ 1200 итер. Валидируемся каждые 5 эпох, чекпоинт каждые 10, лучший
по mDice сохраняется автоматически.
"""
# Оптимайзер — SGD из урока (DeepLab/UNet стандартно с SGD)
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=5e-4)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)

# Распорядок LR — Poly (стандарт mmseg)
param_scheduler = [
    dict(type="PolyLR", eta_min=1e-4, power=0.9, begin=0, end=100, by_epoch=True),
]

# Циклы
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Хуки. Чекпоинт сохраняет лучший по mDice автоматически.
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=10,
                    save_best="mDice", rule="greater", max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", interval=20, draw=True),
)
