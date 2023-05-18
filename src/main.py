import cv2
import argparse
from src.utils import apply_image_processing, save_statistics_to_csv, calculate_contour_statistics


def main(input_image_path, output_image_path, output_csv_path):
    orig_img = cv2.imread(input_image_path)

    img_rgb, contours = apply_image_processing(orig_img)

    contour_statistics = calculate_contour_statistics(contours)

    # Write the output image
    cv2.imwrite(output_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Save the contour statistics to a CSV
    save_statistics_to_csv(contour_statistics, output_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image_path", type=str)
    parser.add_argument("output_image_path", type=str)
    parser.add_argument("output_csv_path", type=str)
    args = parser.parse_args()

    main(args.input_image_path, args.output_image_path, args.output_csv_path)
