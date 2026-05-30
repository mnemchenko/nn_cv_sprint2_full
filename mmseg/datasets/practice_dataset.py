"""PracticeDataset — датасет для задания Спринта 2 (cat/dog).

Регистрируется в системе mmsegmentation через декоратор @DATASETS.register_module(),
чтобы можно было ссылаться на него в конфигах строкой 'PracticeDataset'.

Формат входа — стандартный для BaseSegDataset:
    <data_root>/img/<split>/<name>.jpg
    <data_root>/labels/<split>/<name>.png   # uint8, значения 0=bg, 1=cat, 2=dog
    <data_root>/splits/<split>.txt          # опц. список stem'ов (по строке)
"""
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PracticeDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "cat", "dog"),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
