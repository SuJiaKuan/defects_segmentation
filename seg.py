import os

import numpy as np
import cv2


def imshow(winname, img):
    cv2.imshow(winname, cv2.resize(img, (960, 540)))


def main():
    clean_dir = "/data/feabries/video_restoration/mazu_7200/images/clean"
    noisy_dir = "/data/feabries/video_restoration/mazu_7200/images/noisy"

    img_idx = 11
    binary_threshold = 50
    kernel_size = 3
    area_threshold = 10

    img_filename = "{}.png".format(str(img_idx).zfill(8))
    img_clean_filepath = os.path.join(clean_dir, img_filename)
    img_noisy_filepath = os.path.join(noisy_dir, img_filename)

    img_clean = cv2.imread(img_clean_filepath)
    img_noisy = cv2.imread(img_noisy_filepath)

    img_diff = cv2.absdiff(img_clean, img_noisy)
    img_diff_gray = np.max(img_diff, axis=2)
    img_diff_blur = cv2.blur(img_diff_gray, (kernel_size, kernel_size))
    _, img_diff_binary = cv2.threshold(
        img_diff_blur,
        binary_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    _, contours, hierarchy = cv2.findContours(
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

    cv2.imwrite("output.png", img_noisy_contours)

    imshow("Clean", img_clean)
    imshow("Noisy", img_noisy)
    imshow("Diff", img_diff)
    imshow("Diff Gray", img_diff_gray)
    imshow("Diff Blur", img_diff_blur)
    imshow("Diff Binary", img_diff_binary)
    imshow("Noisy Contours", img_noisy_contours)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
