"""[summary]
    Take mouse as the example and test different convention method to get contour by OpenCV

    - OTSU threshold to find the contour
"""
import argparse
import logging
from itertools import compress
from multiprocessing import Pool, cpu_count
from pathlib import Path

import yaml

from src.components import DetectionTarget, Video
from src.methods.retinex import automated_msrcr
from src.methods.retinex import multi_scale_retinex_chromaticity_preservation as MSRCP
from src.methods.retinex import multi_scale_retinex_color_restoration as MSRCR
from src.methods.threshold import active_contour_by_threshold, find_contour_by_threshold
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
    parser.add_argument('-s', '--savepath', dest='savepath', help='video save path')
    return parser

def _image_preprocess(img, config, option):
    if option == 'MSRCR':
        img = MSRCR(img, **config['retinex']['MSRCR'])
    elif option == 'autoMSRCR':
        img = automated_msrcr(img, **config['retinex']['auto_MSRCR'])
    elif option == 'MSRCP':
        img = MSRCP(img, **config['retinex']['MSRCP'])
    return img

def _image_get_contour(img, config, option):
    if option == 'threshold':
        return find_contour_by_threshold(img, **config['cv2'], **config['general']['filter'])
    elif option == 'active_contour':
        return active_contour_by_threshold(img, **config['general']['filter'])
    return None

def get_contour_from_video(videopath: str, \
                           frame_idx: int, \
                           config: dict, preprocess_option: str, contour_option: str):
    """basic process to get the contour from scratch"""
    with Video(videopath) as video:
        frame_img = video.read_frame(frame_idx)
        if frame_img is None:
            return None

        frame_img = _image_preprocess(frame_img, config, preprocess_option)
        cnts = _image_get_contour(frame_img, config, contour_option) or None
        if cnts is None:
            return None
        target = DetectionTarget(frame_idx, cnts)
        return target

def main(args: argparse.Namespace):
    """[summary]
    an interface to handle input and parse to different method

    Arguments:
        args {argparse.Namespace} -- parameter parse fomr terminal
    """
    # config
    logger = logging.getLogger(__name__)
    log_handler(logger, logging.getLogger(Video.__class__.__name__))
    logger.info(args)
    with open(CONFIG_FILE) as config_file:
        config = yaml.load(config_file)

    # test single frame from video and the conventional find contour method
    with Video(args.input) as video:

        # multiprocessing calc the contour
        pending_frame_idx = list(range(0, video.frame_count, config['general']['skip_per_nframe']))
        logger.info('process pending frame index: %d', len(pending_frame_idx))
        logger.info('cpu count: %d (will used %d)', cpu_count, (cpu_count*3//4))
        with Pool(processes=(cpu_count*3//4)) as pool:
            # basic contours
            mp_args = zip([args.input]*len(pending_frame_idx), \
                           pending_frame_idx, \
                           [config]*len(pending_frame_idx), \
                           [args.preprocess]*len(pending_frame_idx), \
                           [args.option]*len(pending_frame_idx))
            mp_targets = pool.starmap_async(get_contour_from_video, mp_args)
            mp_targets = mp_targets.get()
            mp_targets = [i for i in mp_targets if i]
            mp_targets = sorted(mp_targets, key=lambda x: x.frame_idx)
            _basic_target_counts = len(mp_targets)

            # interpolate contours
            while True:
                # find the new pending index
                pair_targets = [(mp_targets[i], mp_targets[i+1]) for i in range(len(mp_targets)-1)]
                check_frame_idx = [i.is_shifting(j) for i, j in pair_targets]
                if not any(check_frame_idx):
                    break
                pending_frame_idx = list(compress(pair_targets, check_frame_idx))
                pending_frame_idx = [(i.frame_idx+j.frame_idx)//2 \
                                     for i, j in pending_frame_idx if i.frame_idx+1 < j.frame_idx]

                # multiprocessing calc
                mp_args = zip([args.input]*len(pending_frame_idx), \
                              pending_frame_idx, \
                              [config]*len(pending_frame_idx), \
                              [args.preprocess]*len(pending_frame_idx), \
                              [args.option]*len(pending_frame_idx))
                mp_interpolate_targets = pool.starmap_async(get_contour_from_video, mp_args)
                mp_interpolate_targets = mp_interpolate_targets.get()
                mp_targets += [i for i in mp_interpolate_targets if i]
                mp_targets = sorted(mp_targets, key=lambda x: x.frame_idx)

            logger.info('#contours after interpolate: %d -> %d', \
                        _basic_target_counts, len(mp_targets))
            video.detect_targets += mp_targets
        video.save(args.savepath, draw_cnts=True)

if __name__ == '__main__':
    main(argparser().parse_args())
