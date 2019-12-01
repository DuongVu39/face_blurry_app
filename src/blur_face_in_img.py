import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from skimage import data
from skimage import io as skimage_io
from skimage.feature import Cascade
from skimage.filters import gaussian
import matplotlib.pyplot as plt


# TODO: make a class FaceBlurrer


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
        trained_file = data.lbp_frontal_face_cascade_filename()
        self.detector = Cascade(trained_file)

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
        for obj_bb in objects_detected:
            self.blur_object(obj_bb)

        return self.img


def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Required Args
    parser.add_argument("--input-img", help="Path to input image", type=str,
                        default="")

    # Parse arguments
    args = parser.parse_args()

    return args


def save_result(input_path, img) -> bool:
    output_name = Path("../results") / ("Blurred_" + input_path)
    try:
        skimage_io.imsave(output_name, img)
        return True
    except Exception as e:
        print(e)
        return False


def load_img(input_path):
    if len(input_path) > 0:
        img = skimage_io.imread(Path("data") / input_path)
    else:
        img = data.astronaut()
    return img


if __name__ == "__main__":
    args = parse_arguments()

    img = load_img(args.input_img)
    blurrer = FaceBlurrer()
    img = blurrer.compute(img)
    result = save_result(args.input_img if args.input_img != ''
                         else "default_img.png", img)

    plt.imshow(img)
    plt.show()
    print(result)
