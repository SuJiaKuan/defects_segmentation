import argparse
import glob
import os
import pathlib
from tqdm import tqdm

import cv2

from defects_segmentation.defects import find_defects
from defects_segmentation.io import save_json


LABEL_NAME = "defect"


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


def gen_defects(img_triplets, output_root):
    mkdir_p(output_root)

    for clean_dir, noisy_dir, img_name in tqdm(img_triplets):
        img_clean = cv2.imread(os.path.join(clean_dir, img_name))
        img_noisy = cv2.imread(os.path.join(noisy_dir, img_name))

        contours = find_defects(img_clean, img_noisy)

        label_dict = {
            "imgHeight": img_clean.shape[0],
            "imgWidth": img_clean.shape[1],
            "objects": [{
                "label": LABEL_NAME,
                "polygon": c.reshape(-1, 2).tolist(),
            } for c in contours],
        }

        output_path = os.path.join(
            output_root,
            "{}.json".format(os.path.splitext(img_name)[0]),
        )
        save_json(output_path, label_dict)


def main(args):
    img_triplets = search_image_triplets(args.imgs_dir)
    gen_defects(img_triplets, args.output)

    print("Results saved in {}".format(args.output))


if __name__ == "__main__":
    main(parse_args())
