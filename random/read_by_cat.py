#!/usr/bin/python
import os
from PIL import Image
from shutil import copyfile

import pandas as pd


def get_image_size(filepath):
    with Image.open(filepath) as img:
        width, height = img.size
    print (width, height)
    return width * height

ANNS = "/home/bjafek/Nuro/benj_prac/fathomnet/data/train/annotations.csv"
CATS = [
    "Sebastolobus",
    "Apostichopus leukothele",
    "Sebastes",
    "Sebastes diploproa",
]

OUT_DIR = "/home/bjafek/Nuro/benj_prac/fathomnet/get_image_size/special_cluster2"

for cat in CATS:
    df = pd.read_csv(ANNS)
    df = df[df.label == cat]
    df['size'] = df['path'].apply(get_image_size)
    df.sort_values(by="size", inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)

    print (df)
    out_csv = os.path.join(OUT_DIR, f"{cat}.csv")
    df.to_csv(out_csv, index=False)

    for (idx, row) in df.iterrows():
        out_name = os.path.join(OUT_DIR, f"{cat}_{idx}.png")
        copyfile(row["path"], out_name)

        if idx > 5:
            break

