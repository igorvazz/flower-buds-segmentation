import cv2
import numpy as np
from src.utils import apply_image_processing

def test_apply_image_processing():
    # Load test image
    img = cv2.imread('tests/test_image.jpg')

    # Apply image processing
    output_img, contours = apply_image_processing(img)

    # Perform assertions on the output
    assert isinstance(output_img, np.ndarray)
    assert output_img.shape == img.shape

    # Perform assertions on the detected contours
    assert isinstance(contours, tuple)
    assert len(contours) > 0


