import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple


def adjust_brightness_and_contrast(
    img: np.ndarray, contrast: float = 1.0, brightness: int = 0
) -> np.ndarray:
    """
    Adjust brightness and contrast to make the buds more visible.
    """
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def convert_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Convert image to HSV color space, in order to easier filter white color.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def threshold_hsv(hsv: np.ndarray, lower_white: np.ndarray, upper_white: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, lower_white, upper_white)


def apply_bitwise_and(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=mask)


def convert_to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_threshold(
    gray: np.ndarray, threshold_val: int, max_val: int, threshold_type: int
) -> np.ndarray:
    ret, thresh = cv2.threshold(gray, threshold_val, max_val, threshold_type)
    return thresh


def apply_morphology_ex(
    thresh: np.ndarray, kernel_size: Tuple[int, int], morphology_type: int
) -> np.ndarray:
    """
    Perform opening to remove small noise, using an elliptical kernel.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(thresh, morphology_type, kernel)


def apply_gaussian_blur(img: np.ndarray, blur_size: Tuple[int, int], sigma: int) -> np.ndarray:
    return cv2.GaussianBlur(img, blur_size, sigma)


def find_contours(
    img: np.ndarray, contour_retrieval_mode: int, contour_approximation_method: int
) -> List[np.ndarray]:
    contours, _ = cv2.findContours(img, contour_retrieval_mode, contour_approximation_method)
    return contours

def calculate_contour_statistics(contours: List[np.ndarray]) -> List[Tuple[float, Tuple[int, int]]]:
    contour_statistics = []
    
    for cnt in contours:
        # calculate area
        area = cv2.contourArea(cnt)
        
        # calculate center
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        contour_statistics.append((area, (cX, cY)))
    
    return contour_statistics

def draw_contours(
    img: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """
    The function calculates the average contour area and draws only the contours whose area is less than or 
    equal to the average area + a delta.
    """
    avg_area = np.mean([cv2.contourArea(cnt) for cnt in contours])
    for cnt in contours:
        if cv2.contourArea(cnt) <= avg_area + 1000:
            cv2.drawContours(img, [cnt], -1, color, thickness)
    return img

def save_statistics_to_csv(contour_statistics: List[Tuple[float, Tuple[int, int]]], output_csv_path: str) -> None:
    data = {
        'contour_area': [area for area, _ in contour_statistics],
        'CX': [cx for _, (cx, _) in contour_statistics],
        'CY': [cy for _, (_, cy) in contour_statistics]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)

def apply_image_processing(orig_img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    img = orig_img.copy()
    img = adjust_brightness_and_contrast(img, contrast=1.9, brightness=-127)
    hsv = convert_to_hsv(img)

    # Filtering HSV image to select white color
    lower_white = np.array([0, 0, 255])
    upper_white = np.array([255, 200, 255])
    mask = threshold_hsv(hsv, lower_white, upper_white)
    img = apply_bitwise_and(img, mask)

    # Morphological operations
    gray = convert_to_gray(img)
    thresh = apply_threshold(gray, 200, 255, cv2.THRESH_OTSU)
    opening = apply_morphology_ex(thresh, (9, 9), cv2.MORPH_OPEN)
    opening = apply_gaussian_blur(opening, (5, 5), 2)

    # Find contours
    contours = find_contours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert color style from BGR to RGB
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Draw only the contours whose area is less than or equal to the average area
    img_rgb = draw_contours(img_rgb, contours)

    return img_rgb, contours
