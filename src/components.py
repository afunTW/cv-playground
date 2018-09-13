"""[summary]
Define the component in this project
"""
import abc
import logging
from pathlib import Path
from subprocess import call

import cv2
import numpy as np

from tqdm import tqdm


class InternalInputObject(abc.ABC):
    """internal class for input source

    Attributes:
        src: source data
        src_path: source path
        logger: logging.Logger to log the class message
    """
    def __init__(self, src_path: str = ''):
        self.src = None
        self.src_path = src_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def __enter__(self):
        """read when the instance be declared"""
        self.load()
        return self

    @abc.abstractmethod
    def load(self, src=None):
        """load the data from path or assign directly"""
        raise NotImplementedError

class InternalPanelObject(abc.ABC):
    """internal class for visualize result

    Attributes:
        cnt_color: contour color
        logger: logging.Logger to log the class message
        backend: render backend (e.g. cv2, plt, ...)
    """
    def __init__(self, cnt_color: tuple = (0, 255, 0), backend: str = 'cv2'):
        self.cnt_color = cnt_color
        self.logger = logging.getLogger(self.__class__.__name__)
        self._backend = backend

    @property
    def backend(self):
        """render backend (e.g. cv2, plt, ...)"""
        return self._backend

    def __enter__(self):
        if self._backend == 'cv2':
            self.logger.info('Use backend cv2 to render')
            windows_name = 'show'
            # if self.src.__dict__.get('src_path'):
            #     windows_name = self.src.__dict__.get('src_path')
            cv2.namedWindow(windows_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._backend == 'cv2':
            cv2.destroyAllWindows()

    @abc.abstractmethod
    def show(self):
        """show source data"""
        raise NotImplementedError

class Frame(InternalInputObject):
    """represent as the image or the frame in video sequence"""
    def __init__(self, src_path: str = ''):
        super().__init__(src_path=src_path)
        self.src = None

    @property
    def height(self):
        """image height"""
        return int(self.src.shape[0]) if self.src else 0

    @property
    def width(self):
        """image width"""
        return int(self.src.shape[1]) if self.src else 0

    @property
    def channel(self):
        """image channel"""
        return int(self.src.shape[2]) if self.src and len(self.src.shape) == 3 else 0

    def load(self, src: np.ndarray = None):
        """assign video frame or load image from srcpath"""
        if src is not None:
            self.src = src
        elif self.src_path:
            self.src = cv2.imread(self.src_path)

class FramePanel(InternalPanelObject):
    """single frame or image visulaization"""
    def __init__(self, src: Frame, cnt_color: tuple = (0, 255, 0), backend: str = 'cv2'):
        super().__init__(cnt_color, backend)
        self.frame = src

    def draw_contours(self, cnts: np.ndarray):
        """[summary]
        draw contours

        Arguments:
            cnts {np.ndarray} -- contours
        """
        for cnt_idx, cnt in enumerate(cnts):
            if cnt_idx == 0:
                continue
            pt1, pt2 = tuple(cnts[cnt_idx-1][0]), tuple(cnt[0])
            cv2.line(self.frame.src, pt1, pt2, self.cnt_color, 2, cv2.LINE_AA)

    def show(self):
        cv2.imshow('show', self.frame.src)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            cv2.destroyAllWindows()
        else:
            self.show()

class Video(InternalInputObject):
    """represent as the videoe sequence"""
    def __init__(self, src_path: str = ''):
        super().__init__(src_path=src_path)
        self.cap = None
        self.frame_idx = 0
        self.detect_targets = []

    @property
    def frame_count(self):
        """total number of frames"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0

    @property
    def frame_height(self):
        """frame height"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0

    @property
    def frame_width(self):
        """frame width"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0

    @property
    def fps(self):
        """frame per second"""
        return int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def extend_target_to_each_frame(self, simply: bool = True):
        """extend the detection target to each frame

        Keyword Arguments:
            simply {bool} -- return frame_idx and center if True, 
                             or return the DetectionTarget if False (default: {True})
        """
        if not self.detect_targets:
            return None
        target_idx = 0
        results = []
        for frame_idx in range(self.frame_count):
            target = self.detect_targets[target_idx]
            if target.frame_idx < frame_idx:
                target_idx = min(target_idx+1, self.frame_count)
                target = self.detect_targets[target_idx]
            
            if simply:
                results.append((frame_idx, target.frame_idx, target.center))
            else:
                results.append(target)
        return results

    def load(self, src: cv2.VideoCapture = None):
        """load the video from src_path"""
        if src is not None:
            self.cap = src
        else:
            self.cap = cv2.VideoCapture(self.src_path)

    def save(self, savepath: str, draw_cnts: bool = False):
        """save the video to given path

        Arguments:
            savepath {str} -- save path

        Keyword Arguments:
            draw_cnts {bool} -- whether to draw contour (default: {False})
        """
        if draw_cnts and self.detect_targets:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            resolution = tuple(map(int, (self.frame_width, self.frame_height)))
            video_writer = cv2.VideoWriter(str(savepath), fourcc, self.fps, resolution)
            self.detect_targets = sorted(self.detect_targets, key=lambda x: x.frame_idx)
            target_idx = 0

            for frame_idx in tqdm(range(self.frame_count)):
                frame = self.read_frame(frame_idx)

                # choose the target object
                target = self.detect_targets[target_idx]
                if target.frame_idx < frame_idx:
                    target_idx = min(target_idx+1, len(self.detect_targets)-1)
                    target = self.detect_targets[target_idx]
                for idx, cnt in enumerate(target.cnts):
                    if idx == 0:
                        continue
                    pt1, pt2 = tuple(target.cnts[idx-1][0]), tuple(cnt[0])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'frame ({}/{})'.format(frame_idx+1, self.frame_count), \
                            (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                video_writer.write(frame)
            video_writer.release()
        else:
            self.logger.info('No any modification, copy from source')
            call(['rsync', '-av', self.src_path, savepath])

    def read_frame(self, frame_idx):
        """read frame and check if read success"""
        assert frame_idx < self.frame_count, 'frame idx should be less than %d' % self.frame_count
        self.frame_idx = frame_idx
        self.cap.set(1, self.frame_idx)
        read_success, frame = self.cap.read()
        if read_success:
            return frame
        self.logger.exception('read #%s frame failed', str(self.frame_idx))
        return None

class DetectionTarget():
    """detection target to save the contour and related property as data class"""
    def __init__(self, frame_idx: int, cnts: np.ndarray):
        """
        Arguments:
            frame_idx {int} -- frame index
            cnts {np.ndarray} -- contour
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._frame_idx = frame_idx
        self._cnts = cnts
        self._rect = None
        self._center = None

    @property
    def frame_idx(self):
        """return private frame index"""
        return self._frame_idx

    @property
    def cnts(self):
        """return private contours"""
        return self._cnts

    @property
    def rect(self):
        """return the rect coordinate generated from contours"""
        if not self._rect:
            self._rect = cv2.boundingRect(self._cnts)
        return self._rect

    @property
    def center(self):
        """return the center cooridinate generated from rect"""
        if not self._center:
            moment = cv2.moments(self._cnts)
            coor_x = int(moment['m10'] / moment['m00']) if moment['m00'] else 0
            coor_y = int(moment['m01'] / moment['m00']) if moment['m00'] else 0
            self._center = (coor_x, coor_y)
        return self._center

    def is_shifting(self, target, lower_bound: float, upper_bound: float):
        """check the shifting between two DetectionTrget object by L2 dist

        Arguments:
            target {DetectionTarget} -- target object
            lower bound {float} -- tolerable_shifting_dist
            upper bound {float} -- ignored_shfting_dist
        """
        l2_dist = np.linalg.norm(np.array(self.center) - np.array(target.center))
        return lower_bound < l2_dist < upper_bound
