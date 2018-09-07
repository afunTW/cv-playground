"""[summary]
    Take mouse as the example and test different convention method to get contour by OpenCV

    - OTSU threshold to find the contour
"""
import argparse
import logging
from pathlib import Path

import yaml

from src.methods import find_contour_by_threshold
from src.components import Video, Frame, FramePanel
from src.utils import log_handler

CONFIG_FILE = str(Path(__file__).resolve().parents[0] / 'config.yaml')

def argparser():
    """[summary]

    Returns:
        [argparse.NameSpace] -- the parameter parse from terminal
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='could be video or image')
    parser.add_argument('-o', '--option', dest='option', help='method to handle image')
    return parser

def main(args: argparse.Namespace):
    """[summary]
    an interface to handle input and parse to different method

    Arguments:
        args {argparse.Namespace} -- parameter parse fomr terminal
    """
    # config
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)
    with open(CONFIG_FILE) as config_file:
        config = yaml.load(config_file)

    # test single frame from video and the conventional find contour method
    with Video(args.input) as video:
        frame = Frame()
        frame.load(video.read_frame(0))
        cnts = find_contour_by_threshold(frame.src, **config['cv2'])

        with FramePanel(frame) as panel:
            panel.draw_contours(cnts)
            panel.show()


if __name__ == '__main__':
    main(argparser().parse_args())
