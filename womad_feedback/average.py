from pathlib import Path

import click
import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent


def create_average_image(project_folder: Path):
    filenames = project_folder.glob("**/02 - aligned/*.png")
    filenames = list(filenames)

    # Extract alignment template
    groups = {}
    for filename in filenames:
        _, template = filename.stem.rsplit("-", 1)
        groups.setdefault(template, []).append(filename)

    for template, files in groups.items():
        images = [cv2.imread(f.as_posix()) for f in files]
        average_image = np.mean(images, axis=0).astype(np.uint8)

        image_path = project_folder / f"average-{template}.png"
        cv2.imwrite(image_path.as_posix(), average_image)
        click.echo(f"Saved {image_path}")


def average_images():
    filenames = (ROOT / "output").glob("**/aligned/*2022..png")

    images = []
    binary_images = []
    for filename in filenames:
        img = cv2.imread(filename.as_posix())
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray_img)

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary_images.append(thresh)

    average_image = np.mean(images, axis=0).astype(np.uint8)
    cv2.imwrite((ROOT / "output" / "average.png").as_posix(), average_image)

    average_image_bin = np.mean(binary_images, axis=0).astype(np.uint8)
    cv2.imwrite((ROOT / "output" / "average-bin.png").as_posix(), average_image_bin)

    _, average_tresh = cv2.threshold(average_image_bin, 200, 255, cv2.THRESH_BINARY)
    average_tresh = cv2.cvtColor(average_tresh, cv2.COLOR_BGR2GRAY)
    cv2.imwrite((ROOT / "output" / "average-bin-bw.png").as_posix(), average_tresh)

    # Create a kernel for dilation, you can change its size and shape
    kernel = np.ones((7, 7), np.uint8)  # 3x3 square, you can adjust this size

    inverted_mask = cv2.bitwise_not(average_tresh)
    # Assuming `mask` is your average binary mask image
    dilated_mask = cv2.dilate(
        inverted_mask, kernel, iterations=1
    )  # Increase the number of iterations for more dilation
    cv2.imwrite((ROOT / "output" / "average-mask.png").as_posix(), dilated_mask)
    # dilated_mask = cv2.bitwise_not(dilated_mask)

    for ix, img in enumerate(images):
        _, thresholded_roi = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        # diff_img = cv2.absdiff(thresholded_roi, average_tresh)
        masked_img = cv2.bitwise_or(thresholded_roi, dilated_mask)

        out_file = ROOT / "output" / f"diff-{ix:03d}.png"
        print(out_file)
        cv2.imwrite(out_file.as_posix(), masked_img)


if __name__ == "__main__":
    average_images()
