"""[summary]
    Take mouse as the example and test different convention method to get contour by OpenCV

    - OTSU threshold to find the contour
"""
import argparse
import logging
from pathlib import Path

import yaml

from src.components import Frame, FramePanel, Video
from src.methods.retinex import automated_msrcr
from src.methods.retinex import multi_scale_retinex_color_restoration as MSRCR
from src.methods.retinex import multi_scale_retinex_chromaticity_preservation as MSRCP
from src.methods.threshold import find_contour_by_threshold, active_contour_by_threshold
from src.utils import log_handler

CONFIG_FILE = str(Path(__file__).resolve().parents[0] / 'config.yaml')

def argparser():
    """[summary]

    Returns:
        [argparse.NameSpace] -- the parameter parse from terminal
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='could be video or image')
    parser.add_argument('-o', '--option', dest='option', \
                        choices=['threshold', 'active_contour'], \
                        default='threshold', help='method to handle image')
    parser.add_argument('-p', '--preprocess', dest='preprocess', \
                        choices=['MSRCR', 'autoMSRCR', 'MSRCP'], \
                        help='image preprocessing')
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
        frame_img = video.read_frame(0)
        if args.preprocess == 'MSRCR':
            frame_img = MSRCR(frame_img, **config['retinex']['MSRCR'])
        elif args.preprocess == 'autoMSRCR':
            frame_img = automated_msrcr(frame_img, **config['retinex']['auto_MSRCR'])
        elif args.preprocess == 'MSRCP':
            frame_img = MSRCP(frame_img, **config['retinex']['MSRCP'])
        frame = Frame()
        frame.load(frame_img)

        if args.option == 'threshold':
            cnts = find_contour_by_threshold(frame.src, **config['cv2'], **config['general']['filter'])
        elif args.option == 'active_contour':
            cnts = active_contour_by_threshold(frame.src, **config['general']['filter'])

        with FramePanel(frame) as panel:
            panel.draw_contours(cnts)
            panel.show()

if __name__ == '__main__':
    main(argparser().parse_args())
