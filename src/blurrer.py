from abc import ABC, abstractmethod
from skimage import data
from skimage.feature import Cascade
from skimage.filters import gaussian


class BaseBlurrer(ABC):
    # Abstract class to blur object
    def __init__(self):
        self.img = None
        self.detector = None

    @abstractmethod
    def initialize_detector(self):
        # Output should be a model that can
        pass

    @abstractmethod
    def detect_object(self):
        pass

    @abstractmethod
    def get_object(self, row, col, height, width):
        pass

    @abstractmethod
    def blur_object(self, obj_bb):
        pass

    @abstractmethod
    def compute(self, input_path):
        pass


class FaceBlurrer(BaseBlurrer):
    """Concrete class to blur face from image"""

    def __init__(self):
        super(FaceBlurrer, self).__init__()

    def initialize_detector(self) -> None:

        # Output should be a model that can detect faces
        # Load trained file from module root
        weight_path = data.lbp_frontal_face_cascade_filename()
        self.detector = Cascade(weight_path)

    def detect_object(self):
        w, h = self.img.shape[:-1]
        min_w, min_h = w * 0.01, h * 0.01
        max_w, max_h = w * 0.9, h * 0.9
        # detected contain {'c':int, 'r':int, 'width':int, 'height':int}
        detected = self.detector.detect_multi_scale(img=self.img,
                                                    scale_factor=1.2,
                                                    step_ratio=1,
                                                    min_size=(min_w, min_h),
                                                    max_size=(max_w, max_h))
        return detected

    def get_object(self, row, col, height, width):
        objects = self.img[row:(row + width), col:(col + height)]
        return objects

    def _blur_object(self, object, sigma: float = 9.0):
        if len(object.shape) > 2:
            multichannel = object.shape[-1] > 1
        else:
            multichannel = False
        blurred_obj = gaussian(object, sigma=sigma, multichannel=multichannel)
        blurred_obj = (blurred_obj * 255) // 1
        return blurred_obj

    def blur_object(self, obj_bb) -> None:
        # get the variables:
        row = obj_bb['r']
        col = obj_bb['c']
        width = obj_bb['width']
        height = obj_bb['height']

        # blur face
        objects = self.get_object(row, col, height, width)
        blurred_obj = self._blur_object(objects)
        self.img[row:(row + width), col:(col + height), :] = blurred_obj

    def compute(self, img):
        self.initialize_detector()
        self.img = img

        objects_detected = self.detect_object()
        print("Blurred {} face(s)".format(len(objects_detected)))
        for obj_bb in objects_detected:
            self.blur_object(obj_bb)

        return self.img
