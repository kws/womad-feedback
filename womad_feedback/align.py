import shutil
from pathlib import Path
from typing import Any, NamedTuple

import click
import cv2
import numpy as np
from tqdm import tqdm


class ImageStats(NamedTuple):
    keypoints: Any
    descriptors: Any
    image: Any


class ImageAligner:
    def __init__(self, use_sift=False, use_knn=False) -> None:
        self.use_sift = use_sift
        self.use_knn = use_knn
        self.detector = cv2.SIFT_create() if use_sift else cv2.ORB_create()
        self.bf = cv2.BFMatcher(
            cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING, crossCheck=True
        )
        self.base_images = {}

    def get_image_stats(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return ImageStats(keypoints, descriptors, image)

    def add_base_image(self, name, image):
        self.base_images[name] = self.get_image_stats(image)

    def find_best_match(self, image):
        image_stats = self.get_image_stats(image)

        match_stats = []
        for name, base_image in self.base_images.items():
            if self.use_knn:
                matches = self.bf.knnMatch(
                    base_image.descriptors, image_stats.descriptors, k=2
                )
            else:
                matches = self.bf.match(base_image.descriptors, image_stats.descriptors)
            match_stats.append((name, matches))

        match_stats = sorted(match_stats, key=lambda x: len(x[1]), reverse=True)
        return match_stats[0]

    def align(self, image, base_name):
        base_stats = self.base_images[base_name]
        image_stats = self.get_image_stats(image)

        return self.align_with_stats(image, image_stats, base_stats)

    def align_with_stats(self, image, image_stats, base_stats):
        matches = self.bf.match(base_stats.descriptors, image_stats.descriptors)
        if len(matches) < 4:
            click.secho(f"Not enough matches are found - {len(matches)}/4", color="red")
            return None

        # Sort the matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of points in both images
        pts1 = np.float32(
            [base_stats.keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        pts2 = np.float32(
            [image_stats.keypoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Compute homography
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        # Warp image
        height, width = base_stats.image.shape
        return cv2.warpPerspective(image, H, (width, height))

    def align_session(self, session_folder: Path, overwrite=False):
        raw_folder = session_folder / "01 - raw"
        aligned_folder = session_folder / "02 - aligned"
        if aligned_folder.exists() and not overwrite:
            return None

        image_path_list = list(raw_folder.glob("*front.png"))
        image_path_list.sort()

        if len(image_path_list) == 0:
            click.secho(f"No images found in {session_folder}", color="red")
            return None

        shutil.rmtree(aligned_folder, ignore_errors=True)
        aligned_folder.mkdir(parents=True, exist_ok=True)

        progress = tqdm(image_path_list)
        for image_path in progress:
            progress.set_description(f"{image_path.stem}")
            image = cv2.imread(image_path.as_posix(), 0)
            match_name, _ = self.find_best_match(image)
            aligned_image = self.align(image, match_name)

            cv2.imwrite(
                (aligned_folder / f"{image_path.stem}-{match_name}.png").as_posix(),
                aligned_image,
            )

        return aligned_folder
