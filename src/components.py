"""[summary]
Define the component in this project
"""
import abc
import logging

import cv2
import numpy as np


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
        for cnt_idx, cnt in enumerate(cnts[1:]):
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

    def load(self, src: cv2.VideoCapture = None):
        """load the video from src_path"""
        if src is not None:
            self.cap = src
        else:
            self.cap = cv2.VideoCapture(self.src_path)

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
