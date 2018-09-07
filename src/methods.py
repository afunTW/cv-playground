import cv2
import numpy as np


def find_contour_by_threshold(img: np.ndarray, **params):
    """[summary]
    Assume all image read by cv2 (BGR)

    This method will get the conoutr by threshold 
    
    Arguments:
        img {numpy.ndarray} -- input image
    """
    c_gaussian = params.get('GaussianBlur', None)
    c_gaussian_ksize = c_gaussian.get('ksize', (5, 5)) if c_gaussian else (5, 5)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, c_gaussian_ksize, 0)
    _, threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)[-1]
    return cnts

