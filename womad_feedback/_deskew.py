from pathlib import Path

import numpy as np
from deskew import determine_skew
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate


def deskew(_img):
    image = io.imread(_img)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    print(f"Rotating image by {angle} degrees")
    rotated = rotate(image, angle, resize=True) * 255
    return rotated.astype(np.uint8)


if __name__ == "__main__":
    images = Path("output").glob("*-front.png")
    for image in images:
        deskewed = deskew(image)
        io.imsave(image, deskewed)
