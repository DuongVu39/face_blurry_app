import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Any
from pathlib import Path
from loguru import logger
from skimage.feature import Cascade
from skimage import data as skimage_data
# TODO: find another face detector model to add on


class BaseModel(ABC):

    def __init__(self):
        self.model = None

    @abstractmethod
    def load_weight(self, weight_path: Path) -> Any:
        pass

    @abstractmethod
    def initialize_model(self, weights: Any) -> None:
        pass

    @abstractmethod
    def predict(self, x) -> Iterable[Dict[str, int]]:
        # make prediction on x
        pass


class SkimageCascadeModel(BaseModel):

    def __init__(self):
        super(SkimageCascadeModel, self).__init__()

    def load_weight(self, weight_path: Path) -> Any:
        if weight_path.exists():
            # TODO: Implement loading weight file from local path
            pass
        else:
            logger.warning("No existing path was provided, using default weights")
            weight_path = skimage_data.lbp_frontal_face_cascade_filename()

        return weight_path

    def initialize_model(self, weights: Any=None) -> None:
        if weights is None:  # passing a non existent directory
            weights = self.load_weight(weight_path=Path("fdagkjhkjhlkfytfkjhfg"))
        else:
            weights = self.load_weight(weight_path=Path(weights))

        self.model = Cascade(weights)

    def predict(self, x) -> Iterable[Dict[str, int]]:
        w, h = x.shape[:-1]
        min_w, min_h = w * 0.01, h * 0.01
        max_w, max_h = w * 0.9, h * 0.9

        # detected contain {'c':int, 'r':int, 'width':int, 'height':int}
        detected = self.model.detect_multi_scale(img=x,
                                                 scale_factor=1.2,
                                                 step_ratio=1,
                                                 min_size=(min_w, min_h),
                                                 max_size=(max_w, max_h))
        return detected


class YoloFaceModel(BaseModel):

    def __init__(self):
        """Adapted from https://github.com/sthanhng/yoloface"""
        super(YoloFaceModel, self).__init__()

    def load_weight(self, weight_path: Path):
        if weight_path.exists():
            # TODO: Implement loading weight file from local path
            pass
        else:
            logger.warning("No existing path was provided, using default weights")
            weight_path = './model-weights/yolo3-wider_16000.weights'

        return weight_path

    def initialize_model(self, weights: Any):
        pass

    def predict(self, x):
        pass
