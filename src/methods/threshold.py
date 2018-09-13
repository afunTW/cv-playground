"""implement the conventionl method for finding contour

contours format will be (?, 1, 2) in np.uint8 dtype
"""
import cv2
import numpy as np

from skimage.segmentation import active_contour


def _get_focus_rect(img_shape: tuple, \
                    padding_top=0, \
                    padding_right=0, \
                    padding_bottom=0, \
                    padding_left=0, \
                    return_contour=False):
    """[summary]

    Arguments:
        img_shape {tuple} -- image shape

    Keyword Arguments:
        padding_top {int} -- inner padding (default: {0})
        padding_right {int} -- inner padding (default: {0})
        padding_bottom {int} -- inner padding (default: {0})
        padding_left {int} -- inner padding (default: {0})
    """
    height, width = img_shape[:2]
    left_top = (padding_left, padding_top)
    left_bottom = (padding_left, height - padding_bottom)
    right_top = (width - padding_right, padding_top)
    right_bottom = (width - padding_right, height - padding_bottom)

    if return_contour:
        edge = list(range(left_top[0], right_top[0]))
        edge_top = np.array((edge, [padding_top]*len(edge))).T
        edge = list(range(right_top[1], right_bottom[1]))
        edge_right = np.array(([width - padding_right]*len(edge), edge)).T
        edge = list(range(left_bottom[0], right_bottom[0]))
        edge_bottom = np.array((edge, [height - padding_bottom]*len(edge))).T
        edge = list(range(left_bottom[1], left_top[1], -1))
        edge_left = np.array(([padding_left]*len(edge), edge)).T
        edges = np.concatenate((edge_top, edge_right, edge_bottom, edge_left), axis=0)
        edges = edges[:, np.newaxis, ...]
        edges = edges.astype(np.int32)
        return edges

    return (left_top, right_bottom)

def _calc_iou(rect1: tuple, rect2: tuple):
    """calc iou with two rect

    Arguments:
        rect1 {tuple} -- (left_top, right_bottom)
        rect2 {tuple} -- (left_top, right_bottom)

    Returns:
        [type] -- [description]
    """

    x_min = max(rect1[0][0], rect2[0][0])
    y_min = max(rect1[0][1], rect2[0][1])
    x_max = min(rect1[1][0], rect2[1][0])
    y_max = min(rect1[1][1], rect2[1][1])
    area1 = (rect1[1][0] - rect1[0][0])*(rect1[1][1] - rect1[0][1])
    area2 = (rect2[1][0] - rect2[0][0])*(rect2[1][1] - rect2[0][1])

    if x_min < x_max and y_min < y_max:
        inter_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        return inter_area / float(area1 + area2 - inter_area + 1e-5)
    return None

def _allin_focus_rect(src_cnts: np.ndarray, rect: tuple):
    """check rect of contour whether in the rect contour
    (1, 0 ,-1): (inside, on the edge, outside)

    Arguments:
        src_cnts {np.ndarray} -- image contour
        rect {Tuple[tuple, tuple]} -- rect coordinate (left top / right bottom)
    """
    src_x, src_y, src_w, src_h = cv2.boundingRect(src_cnts)
    src_lt, src_rb = (src_x, src_y), (src_x + src_w, src_y + src_h)

    x_min = max(src_lt[0], rect[0][0])
    y_min = max(src_lt[1], rect[0][1])
    x_max = min(src_rb[0], rect[1][0])
    y_max = min(src_rb[1], rect[1][1])
    rect_area = (src_rb[0]-src_lt[0])*(src_rb[1]-src_lt[1])
    if x_min < x_max and y_min < y_max:
        inter_area = (x_max-x_min)*(y_max-y_min)
        if inter_area == rect_area:
            return True
        return False
    return None

def otsu_mask(img: np.ndarray, \
              gaussian_ksize: tuple = (5, 5)):
    """return the otsu mask for debugging

    Arguments:
        img {np.ndarray} -- process image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, gaussian_ksize, 0)

    _, threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold

def find_contour_by_threshold(img: np.ndarray, \
                              gaussian_ksize: tuple = (5, 5), \
                              padding_top: int = 0, \
                              padding_right: int = 0, \
                              padding_bottom: int = 0, \
                              padding_left: int = 0):
    """Assume all image read by cv2 (BGR)

    This method will get the conoutr by threshold

    Arguments:
        img {numpy.ndarray} -- input image
        gaussian_ksize {tuple} -- filter kernel for gaussian blur
    """
    threshold = otsu_mask(img, gaussian_ksize)
    _, cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter
    if any([padding_top, padding_right, padding_bottom, padding_left]):
        rect = _get_focus_rect(img.shape, padding_top, padding_right, padding_bottom, padding_left)
        cnts = [c for c in cnts if _allin_focus_rect(c, rect)]

    if cnts:
        return sorted(cnts, key=cv2.contourArea)[-1]
    return None

def active_contour_by_threshold(img: np.ndarray, \
                                padding_top: int = 0, \
                                padding_right: int = 0, \
                                padding_bottom: int = 0, \
                                padding_left: int = 0):
    """find contour by skimage

    Arguments:
        img {np.ndarray} -- [description]
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    init_snake = _get_focus_rect(img.shape, \
                                 padding_top, padding_right, padding_bottom, padding_left, True)
    cnts = active_contour(threshold, init_snake[:, 0, :])
    cnts = cnts[:, np.newaxis, ...]
    cnts = cnts.astype(np.uint8)
    return cnts
