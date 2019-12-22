import os
import pytest
import numpy as np
from skimage import data
from face_blurry_app.src.blur_face_in_img import load_img, save_result


def test_load_img():
    """Test function load_img from blur_face_in_img.py"""
    # test if nothing is passed, the default img is called
    image = load_img("")
    assert np.all(image == data.astronaut()), "If nothing is passed, default image should be called"
    # test if an invalid path is passed, Value Error should be called
    with pytest.raises(ValueError):
        load_img("wrong_image.png")
    # test if a valid path is passed, it should return that img.
    os.chdir('../face_blurry_app/')
    image = load_img("dog.png")
    assert image.shape != data.astronaut().shape, "A valid path is passed, return image should not be the default."


def test_save_result():
    pass
