import cv2
import numpy as np
from src.utils import calculate_contour_statistics

def test_calculate_contour_statistics():
    # Create a test image with known contours
    img = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.circle(img, (150, 150), 20, 255, -1)
    cv2.circle(img, (700, 700), 30, 255, -1)
    cv2.circle(img, (300, 700), 15, 255, -1)

    # Find contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Call the calculate_contour_statistics function
    contour_statistics = calculate_contour_statistics(contours)

    # Perform assertions on the contour statistics
    assert len(contour_statistics) == 3
    assert isinstance(contour_statistics[0], tuple)