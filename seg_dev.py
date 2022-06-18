import argparse
import os

import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Development script for semantic segmentation dataset "
                    "generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "img_clean",
        type=str,
        help="Input clean image path",
    )
    parser.add_argument(
        "img_noisy",
        type=str,
        help="Input noisy image path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.png",
        help="Output path",
    )

    args = parser.parse_args()

    return args


def imshow(winname, img):
    cv2.imshow(winname, cv2.resize(img, (960, 540)))


def main(args):
    binary_threshold = 50
    kernel_size = 3
    area_threshold = 10

    img_clean = cv2.imread(args.img_clean)
    img_noisy = cv2.imread(args.img_noisy)

    img_diff = cv2.absdiff(img_clean, img_noisy)
    img_diff_gray = np.max(img_diff, axis=2)
    img_diff_blur = cv2.blur(img_diff_gray, (kernel_size, kernel_size))
    _, img_diff_binary = cv2.threshold(
        img_diff_blur,
        binary_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    contours, hierarchy = cv2.findContours(
        img_diff_binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [c for c in contours if cv2.contourArea(c) >= area_threshold]

    img_noisy_contours = cv2.drawContours(
        img_noisy.copy(),
        contours,
        -1,
        (0, 255, 0),
        3,
    )
    mask = cv2.drawContours(
        np.zeros_like(img_noisy),
        contours,
        -1,
        (255, 255, 255),
        -1,
    )

    cv2.imwrite(args.output, img_noisy_contours)

    imshow("Clean", img_clean)
    imshow("Noisy", img_noisy)
    imshow("Diff", img_diff)
    imshow("Diff Gray", img_diff_gray)
    imshow("Diff Blur", img_diff_blur)
    imshow("Diff Binary", img_diff_binary)
    imshow("Noisy Contours", img_noisy_contours)
    imshow("Mask", mask)
    cv2.waitKey(0)


if __name__ == "__main__":
    main(parse_args())
