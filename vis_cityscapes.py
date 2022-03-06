import argparse

import cv2
import cmapy
import numpy as np
import imutils

from defects_segmentation.io import load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for Cityscapes data visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Input image path",
    )
    parser.add_argument(
        "label",
        type=str,
        help="Input label JSON path",
    )

    args = parser.parse_args()

    return args


def main(args):
    img = cv2.imread(args.image)
    label_polygons = load_json(args.label)["objects"]

    labels = set([x["label"] for x in label_polygons])
    cmap = cmapy.cmap("gist_ncar").reshape(-1, 3).tolist()
    label_cmap = {l: cmap[i * 10] for i, l in enumerate(labels)}

    drawn = img.copy()
    for label_polygon in label_polygons:
        label = label_polygon["label"]
        polygon = np.array(label_polygon["polygon"])

        drawn = cv2.polylines(
            drawn,
            [polygon],
            True,
            color=label_cmap[label],
            thickness=3,
        )

    cv2.imshow("Drawn", imutils.resize(drawn, width=800))
    cv2.waitKey(0)


if __name__ == "__main__":
    main(parse_args())
