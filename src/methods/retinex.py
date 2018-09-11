"""Retinex algorithm
fork and modified from https://github.com/dongb5/Retinex

- Single Scale Retinex (SSR)
- Multi Scale Retinex (MSR)
- Multi Scale Retinex Color Restoration (MSRCR)
- automated MSRCR
- Multi Scale Retinex with Chromaticity Preservation (MSRCP)
"""

import numpy as np
import cv2

def single_scale_retinex(img, sigma):
    """Single Scale Retinex

    Arguments:
        img {np.ndarray} -- process image
        sigma {int} -- [description]

    Returns:
        retinex {np.ndarray} -- float64, needs to rescale to 0~255
    """
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multi_scale_retinex(img, sigma_list):
    """Multi Scale Retinex

    Arguments:
        img {np.ndarray} -- process image
        sigma_list {list of int} -- list of sigma for SSR

    Returns:
        retinex {np.ndarray} -- float64, needs to rescale to 0~255
    """
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def color_restoration(img, alpha, beta):
    """color restoration

    Arguments:
        img {np.ndarray} -- process image
        alpha {float} -- [description]
        beta {float} -- [description]

    Returns:
        img_color_restoration {np.ndarray} -- float64
    """

    img_sum = np.sum(img, axis=2, keepdims=True)
    img_color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return img_color_restoration

def simple_color_balance(img, low_clip, high_clip):
    """simple color balance

    Arguments:
        img {np.ndarray} -- process image
        low_clip {float} -- [description]
        high_clip {float} -- [description]

    Returns:
        img {np.ndarray} -- same dtype with input img
    """

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for uni, count in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = uni
            if float(current) / total < high_clip:
                high_val = uni
            current += count

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img

def multi_scale_retinex_color_restoration(img, \
                                          sigma_list, \
                                          gain, bias, \
                                          alpha, beta, \
                                          low_clip, high_clip):
    """multi_scale_retinex_color_restoration

    Arguments:
        img {np.ndarray} -- process image
        sigma_list {list of int} -- list of sigma for SSR
        gain {float} -- gain
        bias {float} -- bias
        alpha {float} -- parameter for color restoration
        beta {float} -- parameter for color restoration
        low_clip {float} -- parameter for color balance
        high_clip {float} -- parameter for color balance

    Returns:
        [type] -- [description]
    """

    img = np.float64(img) + 1.0

    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = gain * (img_retinex * img_color + bias)

    # basic MSRCR formula
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simple_color_balance(img_msrcr, low_clip, high_clip)
    return img_msrcr

def automated_msrcr(img, sigma_list):
    """automated to find the experiment parameter by min/max color

    Arguments:
        img {np.ndarray} -- process image
        sigma_list {list of int} -- list of sigma for SSR

    Returns:
        img_retinex {np.ndarray} -- [description]
    """

    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, counts = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for uni, count in zip(unique, counts):
            if uni == 0:
                zero_count = count
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for uni, count in zip(unique, counts):
            if uni < 0 and count < zero_count * 0.1:
                low_val = uni / 100.0
            if uni > 0 and count < zero_count * 0.1:
                high_val = uni / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)
    return img_retinex

def multi_scale_retinex_chromaticity_preservation(img, sigma_list, low_clip, high_clip):
    """multi_scale_retinex_chromaticity_preservation
    based on original channel to refine

    Arguments:
        img {np.ndarray} -- process image
        sigma_list {list of int} -- list of sigma for SSR
        low_clip {float} -- [description]
        high_clip {float} -- [description]

    Returns:
        [type] -- [description]
    """

    img = np.float64(img) + 1.0
    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multi_scale_retinex(intensity, sigma_list)
    intensity = np.expand_dims(intensity, 2)

    retinex = np.expand_dims(retinex, 2)
    intensity1 = simple_color_balance(retinex, low_clip, high_clip)
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)
    for axis_y in range(img_msrcp.shape[0]):
        for axis_x in range(img_msrcp.shape[1]):
            max_pixel = np.max(img[axis_y, axis_x])
            min_pixel = np.minimum(256.0 / max_pixel, \
                                   intensity1[axis_y, axis_x, 0] / intensity[axis_y, axis_x, 0])
            img_msrcp[axis_y, axis_x, 0] = min_pixel * img[axis_y, axis_x, 0]
            img_msrcp[axis_y, axis_x, 1] = min_pixel * img[axis_y, axis_x, 1]
            img_msrcp[axis_y, axis_x, 2] = min_pixel * img[axis_y, axis_x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp
