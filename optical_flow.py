"""Tracking by optical flow

- Lucas-Kanade method: sparse optical flow
- Gunner Farneback: dense optical flow
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.components import Video
from src.utils import func_profile, log_handler

CONFIG_FILE = str(Path(__file__).resolve().parents[0] / 'config.yaml')

def argparser():
    """
    Returns:
        [argparse.NameSpace] -- the parameter parse from terminal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-o', '--option', dest='option', \
                        choices=['LK'], default='LK', \
                        help='choose the optical flow')
    return parser

@func_profile
def main(args: argparse.Namespace):
    """optical flow interface

    Arguments:
        args {argparse.Namespace} -- parameter parse fomr terminal
    """
    # config
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)
    with open(CONFIG_FILE) as config_file:
        config = yaml.load(config_file)

    # demo code for optical flow testing
    with Video(args.video) as video:
        prev_frame = video.read_frame(0)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # limit the feature area
        pos_0 = cv2.goodFeaturesToTrack(prev_gray[:300, :400], mask=None, **config['shitomasi'])
        mask = np.zeros_like(prev_frame)

        while True:
            frame_img = video.read_frame(video.frame_idx+1)
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

            # calc optical flow
            pos_1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, \
                                                          pos_0, None, **config['lk'])
            good_new = pos_1[status == 1]
            good_old = pos_0[status == 1]

            for new, old in zip(good_new, good_old):
                pt1, pt2 = tuple(new.ravel()), tuple(old.ravel())
                mask = cv2.line(mask, pt1, pt2, (0, 255, 0), 2)
                frame_img = cv2.circle(frame_img, pt1, 5, (0, 0, 255), -1)
            img = cv2.add(frame_img, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(1)
            if k in [27, ord('q')]:
                cv2.destroyAllWindows()
                break
            prev_gray = frame_gray.copy()
            pos_0 = good_new.reshape(-1, 1, 2)

if __name__ == '__main__':
    main(argparser().parse_args())
