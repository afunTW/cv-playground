"""[summary]
    Take mouse as the example and test different convention method to get contour by OpenCV

    - OTSU threshold to find the contour
"""
import argparse
import logging
import pickle
from itertools import compress
from multiprocessing import Pool, cpu_count
from pathlib import Path

import yaml

import pandas as pd
import numpy as np
from src.components import DetectionTarget, Video, Frame, FramePanel
from src.methods.retinex import automated_msrcr
from src.methods.retinex import multi_scale_retinex_chromaticity_preservation as MSRCP
from src.methods.retinex import multi_scale_retinex_color_restoration as MSRCR
from src.methods.threshold import active_contour_by_threshold
from src.methods.threshold import find_contour_by_threshold
from src.methods.threshold import otsu_mask
from src.utils import log_handler, func_profile

CONFIG_FILE = str(Path(__file__).resolve().parents[0] / 'config.yaml')

def argparser():
    """[summary]

    Returns:
        [argparse.NameSpace] -- the parameter parse from terminal
    """
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='subparser')

    # main parser
    parser.add_argument('-i', '--input', dest='input', help='could be video or image')
    parser.add_argument('-o', '--option', dest='option', \
                        choices=['threshold', 'active_contour'], \
                        default='threshold', help='method to handle image')
    parser.add_argument('-p', '--preprocess', dest='preprocess', \
                        choices=['OTSU', 'MSRCR', 'autoMSRCR', 'MSRCP'], \
                        help='image preprocessing')
    parser.add_argument('-s', '--savepath', dest='savepath', help='video save path')

    # subparser - demo mode
    demo_parser = subparser.add_parser('demo')
    demo_parser.add_argument('--frame', dest='frame_idx', default=0, type=int)
    demo_parser.add_argument('--mask', dest='mask', \
                             action='store_true', \
                             help='chekc the mask after image preprocess')

    parser.set_defaults(mask=False)
    return parser

def _image_preprocess(img, config, option):
    if option == 'OTSU':
        img = otsu_mask(img, **config['cv2'])
        img = img[..., np.newaxis]
        img = np.concatenate((img, img, img), axis=-1)
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
        cnts = _image_get_contour(frame_img, config, contour_option)
        if cnts is None:
            return None
        return (frame_idx, cnts)

def check_contour(videopath: str, frame_idx: int, config: dict, \
                  preoprocess_option: str, contour_option: str, return_mask: bool = False):
    """demo and check one frame in video if contour set is right"""
    with Video(videopath) as video:
        frame_img = video.read_frame(frame_idx)
        if frame_img is None:
            print('frame_img is None')
            return
        frame_img = _image_preprocess(frame_img, config, preoprocess_option)
        if return_mask:
            frame_img = _image_preprocess(frame_img, config, 'OTSU')
        cnts = _image_get_contour(frame_img, config, contour_option)
        frame = Frame()
        frame.load(frame_img)
        with FramePanel(frame) as panel:
            if cnts is not None:
                panel.draw_contours(cnts)
            panel.show()

@func_profile
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

    # demo mode to check if contour is getting right ?
    if args.subparser == 'demo':
        check_contour(args.input, args.frame_idx, config, \
                      args.preprocess, args.option, args.mask)
        return

    # test single frame from video and the conventional find contour method
    with Video(args.input) as video:

        # multiprocessing calc the contour
        pending_frame_idx = list(range(0, video.frame_count, config['general']['skip_per_nframe']))
        logger.info('process pending frame index: %d', len(pending_frame_idx))
        logger.info('cpu count: %d (will used %d)', cpu_count(), (cpu_count()*3//4))
        with Pool(processes=(cpu_count()*3//4)) as pool:
            # basic contours
            mp_args = zip([args.input]*len(pending_frame_idx), \
                           pending_frame_idx, \
                           [config]*len(pending_frame_idx), \
                           [args.preprocess]*len(pending_frame_idx), \
                           [args.option]*len(pending_frame_idx))
            mp_targets = pool.starmap_async(get_contour_from_video, mp_args)
            mp_targets = mp_targets.get()
            mp_targets = [DetectionTarget(*i) for i in mp_targets if i]
            mp_targets = sorted(mp_targets, key=lambda x: x.frame_idx)
            _basic_target_counts = len(mp_targets)

            # interpolate contours
            _check_frame_len = 0
            while True:
                # find the new pending index
                pair_targets = [(mp_targets[i], mp_targets[i+1]) for i in range(len(mp_targets)-1)]
                lower_bound = config['general']['tolerable_shifting_dist']
                upper_bound = config['general']['ignored_shifting_dist']
                check_frame_idx = [i.is_shifting(j, lower_bound, upper_bound) \
                                   for i, j in pair_targets]
                if len(check_frame_idx) == _check_frame_len:
                    break
                _check_frame_len = len(check_frame_idx)
                logger.info('check_frame_idx len=%d (all=%d), any=%s', \
                            len(check_frame_idx), video.frame_count, any(check_frame_idx))
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
                mp_targets += [DetectionTarget(*i) for i in mp_interpolate_targets if i]
                mp_targets = sorted(mp_targets, key=lambda x: x.frame_idx)

            logger.info('#contours after interpolate: %d -> %d', \
                        _basic_target_counts, len(mp_targets))
            video.detect_targets += mp_targets

        # save video
        video_savepath = Path('outputs') / args.savepath
        if not video_savepath.parent.exists():
            video_savepath.parent.mkdir(parents=True)
        video.save(str(video_savepath), draw_cnts=True)

        # save path
        detect_target_per_frame = video.extend_target_to_each_frame(simply=True)
        label = ['frame_idx', 'calc_frame_idx', 'center']
        path_savepath = video_savepath.parent / '{}_path.csv'.format(video_savepath.stem)
        df_path = pd.DataFrame(detect_target_per_frame, columns=label)
        df_path.to_csv(str(path_savepath))

        # save contour
        detect_target_data = {i.frame_idx: i.cnts for i in video.detect_targets}
        cnts_savepath = video_savepath.parent / '{}_cnts.pkl'.format(video_savepath.stem)
        with open(cnts_savepath, 'wb') as f:
            pickle.dump(detect_target_data, f)

if __name__ == '__main__':
    main(argparser().parse_args())
