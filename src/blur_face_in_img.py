import argparse
from pathlib import Path
from skimage import data
from skimage import io as skimage_io
from blurrer import FaceBlurrer
import matplotlib.pyplot as plt


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
    output_name = Path("results") / ("Blurred_" + input_path)
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
