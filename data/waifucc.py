from waifuc.action import HeadCountAction, AlignMinSizeAction
from waifuc.export import SaveExporter
from waifuc.source import DanbooruSource

if __name__ == "__main__":
    source = DanbooruSource(["kafka_(star rail)", "solo"])
    source.attach(
        # only 1 head,
        HeadCountAction(1),
        # if shorter side is over 640, just resize it to 640
        AlignMinSizeAction(640),
    )[
        :10
    ].export(  # only first 10 images
        # save images (with meta information from danbooru site)
        SaveExporter("/data/kafka_waifu_dataset")
    )
