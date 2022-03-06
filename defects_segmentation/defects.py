import cv2
import numpy as np


def find_defects(
    img_clean,
    img_noisy,
    blur_size=3,
    binary_threshold=50,
    area_threshold=10,
):
    img_diff = cv2.absdiff(img_clean, img_noisy)
    img_diff_gray = np.max(img_diff, axis=2)
    img_diff_blur = cv2.blur(img_diff_gray, (blur_size, blur_size))
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

    return contours
