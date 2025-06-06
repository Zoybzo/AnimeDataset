import os

from waifuc.action import HeadCountAction, AlignMinSizeAction
from waifuc.export import SaveExporter
from waifuc.source import DanbooruSource

from utils.get_path import HOME

if __name__ == "__main__":
    source = DanbooruSource(["kafka_(star rail)", "solo"])
    save_path = os.path.join(HOME, "Datasets/kafka_waifu_dataset")
    source.attach(
        # only 1 head,
        HeadCountAction(1),
        # if shorter side is over 640, just resize it to 640
        AlignMinSizeAction(640),
    )[:].export(SaveExporter(save_path))
