import argparse
import glob
import os
import pathlib
import random

import cv2
import numpy as np
from tqdm import tqdm

from defects_segmentation.defects import find_defects
from defects_segmentation.io import save_json


LABEL_NAME = "defect"
LABEL_ID = 1

SPLITS = (
    "train",
    "val",
    "test",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Application for semantic segmentation dataset generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "imgs_dir",
        type=str,
        help="Input images root directory",
    )
    parser.add_argument(
        "-v",
        "--val",
        type=float,
        default=0.1,
        help="Ratio of validation set",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=float,
        default=0.1,
        help="Ratio of test set",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output directory",
    )

    args = parser.parse_args()

    return args


def mkdir_p(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def search_images(imgs_dir):
    return [
        os.path.basename(p)
        for p in glob.glob(os.path.join(imgs_dir, "*.png"))
    ]


def search_image_triplets(imgs_root):
    clean_dir = os.path.join(imgs_root, "clean")
    noisy_dir = os.path.join(imgs_root, "noisy")

    img_clean_names = search_images(clean_dir)
    img_noisy_names = search_images(noisy_dir)
    img_names = sorted(list(set(img_clean_names) & set(img_noisy_names)))

    return [(clean_dir, noisy_dir, n) for n in img_names]


def split_dataset(img_triplets, splits, split_ratios):
    triplet_indices = list(range(len(img_triplets)))
    random.shuffle(triplet_indices)

    dataset = {s: [] for s in splits}

    triplets_len = len(triplet_indices)
    start_idx = 0
    for split_idx, (split, split_ratio) \
            in enumerate(zip(splits, split_ratios)):
        split_len = int(triplets_len * split_ratio)
        if split_idx == len(splits) - 1:
            split_triplet_indices = triplet_indices[start_idx:]
        else:
            split_triplet_indices = \
                triplet_indices[start_idx:start_idx+split_len]

        for triplet_idx in split_triplet_indices:
            dataset[split].append(img_triplets[triplet_idx])

        start_idx = start_idx + split_len

    return dataset


def gen_defects(img_triplets, output_dir):
    mkdir_p(output_dir)

    for clean_dir, noisy_dir, img_name in tqdm(img_triplets):
        img_clean = cv2.imread(os.path.join(clean_dir, img_name))
        img_noisy = cv2.imread(os.path.join(noisy_dir, img_name))

        contours = find_defects(img_clean, img_noisy)

        img_id = os.path.splitext(img_name)[0]

        # Collect and save annotation data.
        label_dict = {
            "imgHeight": img_clean.shape[0],
            "imgWidth": img_clean.shape[1],
            "objects": [{
                "label": LABEL_NAME,
                "polygon": c.reshape(-1, 2).tolist(),
            } for c in contours],
        }
        save_json(
            os.path.join(output_dir, "{}_polygons.json".format(img_id)),
            label_dict,
        )

        # Collect and save annotation data.
        label_img = cv2.drawContours(
            np.zeros_like(img_noisy),
            contours,
            -1,
            (LABEL_ID, LABEL_ID, LABEL_ID),
            -1,
        )
        cv2.imwrite(
            os.path.join(output_dir, "{}_labelIds.png".format(img_id)),
            label_img,
        )


def gen_dataset(dataset, output_root):
    for split, img_triplets in dataset.items():
        print("Generating for {} split".format(split))
        output_dir = os.path.join(output_root, "gtFine", LABEL_NAME, split)
        gen_defects(img_triplets, output_dir)


def main(args):
    split_ratios = (
        (1.0 - args.val - args.test),
        args.val,
        args.test,
    )

    img_triplets = search_image_triplets(args.imgs_dir)
    dataset = split_dataset(img_triplets, SPLITS, split_ratios)
    gen_dataset(dataset, args.output)

    print("Results saved in {}".format(args.output))


if __name__ == "__main__":
    main(parse_args())
