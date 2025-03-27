import os

# %pip install cheesechaser #>0.2.0
from cheesechaser.datapool import Danbooru2024DataPool
from cheesechaser.query import DanbooruIdQuery

from utils.get_path import HOME

pool = Danbooru2024DataPool()
# my_waifu_ids = DanbooruIdQuery(['surtr_(arknights)', 'solo'])
# above is only available when Danbooru is accessible, if not, use following:
import pandas as pd


target_dir = os.path.join(HOME, "Downloads/dataset/danbooru/")
# read parquet file
df = pd.read_parquet(
    os.path.join(target_dir, "metadata.parquet"), columns=["id", "tag_string"]
)  # read only necessary columns

# surtr_(arknights) -> gets interpreted as regex so we need to escape the brackets
subdf = df[
    df["tag_string"].str.contains("surtr_\\(arknights\\)")
    & df["tag_string"].str.contains("solo")
]
ids = subdf.index.tolist()
print(ids[:5])  # check the first 5 ids

dst_dir = os.path.join(target_dir, "surtr")
# download danbooru images with surtr+solo, to directory /data/exp2_surtr
pool.batch_download_to_directory(
    resource_ids=ids,
    dst_dir=dst_dir,
    max_workers=12,
)
