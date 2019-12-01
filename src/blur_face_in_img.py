import argparse
from skimage import data
from skimage import io as skimage_io
from skimage.feature import Cascade
from skimage.filters import gaussian

# TODO: handle black and  white images (gaussian)

def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Required Args
    parser.add_argument("input_img", help="Path to input image", type=str)
    parser.add_argument("output", help="Path to output image", type=str)

    # Parse arguments
    args = parser.parse_args()

    return args


def app_setup() -> tuple:
    # Load trained file from module root
    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(trained_file)
    astronaut_img = data.astronaut()
    img = skimage_io.imread(astronaut_img)
    return detector, img


def detect_face(detector, img):
    # detected contain {'c':int, 'r':int, 'width':int, 'height':int}
    detected = detector.detect_multi_scale(img=img, scale_factor=1.2,
                                           step_ratio=1, min_size=(60, 60), max_size=(123, 123))
    return detected


def get_face(img, row, col, height, width):
    face = img[row:row+height, col:col+width]
    return face


def _blur_face(face, sigma: float=1.0):
    blurred_face = gaussian(face, sigma=sigma, multichannel=True)

    return blurred_face


def blur_face(img, face_bb):

    # get the variables:
    row = face_bb['r']
    col = face_bb['c']
    width = face_bb['width']
    height = face_bb['height']

    # blur face
    face = get_face(img, row, col, height, width)
    blurred_face = _blur_face(face)
    img[row:row+height, col:col+width] = blurred_face
    return img


if __name__ == "__main__":
    args = parse_arguments()

    detector, img = app_setup()
    faces_detected = detect_face(detector, img)
    for face_bb in faces_detected:
        img = blur_face(img, face_bb)
