import shutil
import traceback
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from womad_feedback.align import ImageAligner
from womad_feedback.average import crop_region
from womad_feedback.src_images import IMAGE_PATH, IMAGES

BOX_SIZE = 15
BOX_CROP = 15
NUMBERS_WIDTH = 600
NUMBERS_HEIGHT = 40
NUMBERS_OFFSETS = [-280, -125, 2, 135, 270]


def region_definitions():
    with IMAGE_PATH.joinpath("regions.yaml").open("rt") as f:
        questions = yaml.safe_load(f)["questions"]

    for _, template_def in questions.items():
        for q, q_def in template_def.items():
            if "numbers" in q_def:
                pos = q_def["numbers"]["pos"]
                q_def["boxes"] = boxes = {}
                boxes["Disagree"] = dict(
                    pos=(pos[0] - 440, pos[1]), answer="disagree", width=100
                )
                for i in range(1, 6):
                    x = pos[0] + NUMBERS_OFFSETS[i - 1]
                    boxes[f"{i}"] = dict(pos=(x, pos[1]), answer=i)

                boxes["Agree"] = dict(
                    pos=(pos[0] + 400, pos[1]), answer="agree", width=100
                )

    return questions


def radial_weights(image_shape, center=None, max_weight=1.0):
    """
    Generate a 2D radial weight matrix for an image of given shape.
    """
    if center is None:
        center = (image_shape[1] // 2, image_shape[0] // 2)

    x, y = np.ogrid[: image_shape[0], : image_shape[1]]
    distance_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Normalize distances to be in the range [0, 1]
    max_distance = np.sqrt(center[1] ** 2 + center[0] ** 2)
    normalized_distance = distance_from_center / max_distance

    # Generate radial weights. Closer to center means higher weight.
    weights = max_weight * (1 - normalized_distance)
    return weights


def _gscale(image):
    """Transform image to gray scale"""
    if isinstance(image, Path):
        image = image.as_posix()
    if isinstance(image, str):
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class RegionExtractor:
    def __init__(self, mask_size=7):
        self.defintions = region_definitions()
        self.templates = {template.stem: _gscale(template) for template in IMAGES}
        self.aligner = ImageAligner(use_sift=True, use_knn=True)

        self.template_regions = {}
        self.template_masks = {}

        for template_name, template in self.templates.items():
            self.template_regions.update(
                self.get_image_regions(template, template_name)
            )

        for region_name, image_region in self.template_regions.items():
            self.aligner.add_base_image(region_name, image_region)

        for region_name, image_region in self.template_regions.items():
            _, average_tresh = cv2.threshold(image_region, 200, 255, cv2.THRESH_BINARY)

            # Create a kernel for dilation, you can change its size and shape
            kernel = np.ones((mask_size, mask_size), np.uint8)

            inverted_mask = cv2.bitwise_not(average_tresh)
            # Assuming `mask` is your average binary mask image
            dilated_mask = cv2.dilate(
                inverted_mask, kernel, iterations=1
            )  # Increase the number of iterations for more dilation

            self.template_masks[region_name] = dilated_mask

    def markup_image_regions(self, template_name, boundary_size=35):
        assert (
            template_name in self.templates
        ), f"Unknown template {template_name}. Valid templates are {list(self.templates.keys())}"
        template = self.templates[template_name]

        regions = self.defintions[int(template_name)]
        for _, question_defintion in regions.items():
            if "boxes" in question_defintion:
                for _, box in question_defintion["boxes"].items():
                    x, y = box["pos"]
                    template = cv2.rectangle(
                        template,
                        (x - boundary_size, y - boundary_size),
                        (x + boundary_size, y + boundary_size),
                        0,
                        2,
                    )
        return template

    def get_answer_defintion(self, id_string):
        template_name, q, v = id_string.split("_")
        regions = self.defintions[int(template_name)]
        question = regions[q]
        boxes = {str(k): v for k, v in question["boxes"].items()}
        return boxes[v]

    def get_image_regions(self, image, template_name):
        image_regions = {}

        regions = self.defintions[int(template_name)]
        for question_id, question_defintion in regions.items():
            if "boxes" in question_defintion:
                for box_id, box in question_defintion["boxes"].items():
                    x, y = box["pos"]
                    image_regions[
                        f"{template_name}_{question_id}_{box_id}"
                    ] = crop_region(
                        image,
                        x - BOX_SIZE,
                        y - BOX_SIZE,
                        x + BOX_SIZE,
                        y + BOX_SIZE,
                        pad_value=255,
                    )

        return image_regions

    def get_all_question_ids(self):
        for template_name, template in self.defintions.items():
            for question_name, question in template.items():
                for box_name, _ in question["boxes"].items():
                    yield template_name, question_name, f"{box_name}"

    def get_all_question_names(self):
        for template_name, template in self.defintions.items():
            for _, question in template.items():
                for box_name, _ in question["boxes"].items():
                    yield template_name, question["question"], f"{box_name}"

    def extract_regions(self, image_path: Path, template_name=None, align=True):
        if not template_name:
            _, template_name = image_path.stem.rsplit("-", 1)
        image = _gscale(image_path)
        image_regions = self.get_image_regions(image, template_name)

        regions = {}
        for region_name, image_region in image_regions.items():
            if align:
                try:
                    aligned_image = self.aligner.align(image_region, region_name)
                except Exception as e:
                    tb = traceback.format_exc()
                    click.secho(
                        f"Failed to align {region_name}: {e}\n{tb}", color="red"
                    )
                    continue
            else:
                aligned_image = image_region

            template_image = self.template_regions[region_name]
            mask = self.template_masks[region_name]
            masked_image = np.bitwise_or(aligned_image, mask)
            _, masked_image = cv2.threshold(masked_image, 200, 255, cv2.THRESH_BINARY)
            masked_image = np.invert(masked_image)

            cy, cx = (masked_image.shape[1] // 2, masked_image.shape[0] // 2)
            masked_image = crop_region(
                masked_image, cx - BOX_CROP, cy - BOX_CROP, cx + BOX_CROP, cy + BOX_CROP
            )

            weights = radial_weights((masked_image.shape))
            masked_image = weights * masked_image

            cy, cx = (aligned_image.shape[1] // 2, aligned_image.shape[0] // 2)
            regions[region_name] = np.hstack(
                [
                    crop_region(
                        aligned_image,
                        cx - BOX_CROP,
                        cy - BOX_CROP,
                        cx + BOX_CROP,
                        cy + BOX_CROP,
                    ),
                    crop_region(
                        template_image,
                        cx - BOX_CROP,
                        cy - BOX_CROP,
                        cx + BOX_CROP,
                        cy + BOX_CROP,
                    ),
                    crop_region(
                        self.template_masks[region_name],
                        cx - BOX_CROP,
                        cy - BOX_CROP,
                        cx + BOX_CROP,
                        cy + BOX_CROP,
                    ),
                    masked_image,
                ]
            ), np.sum(masked_image)

        return regions

    def extract_session(self, session_folder, overwrite=False, weight_threshold=20_000):
        aligned_folder = session_folder / "02 - aligned"
        extracted_folder = session_folder / "03 - extracted"
        if extracted_folder.exists() and not overwrite:
            return None

        image_path_list = list(aligned_folder.glob("*.png"))
        image_path_list.sort()

        if len(image_path_list) == 0:
            click.secho(f"No images found in {session_folder}", color="red")
            return None

        shutil.rmtree(extracted_folder, ignore_errors=True)
        extracted_folder.mkdir(parents=True, exist_ok=True)

        progress = tqdm(image_path_list)

        all_questions = list(self.get_all_question_ids())
        all_questions_flat = [f"{x[0]}_{x[1]}_{x[2]}" for x in all_questions]
        results = {}
        for image_path in progress:
            results.setdefault(image_path.stem, {k: "" for k in all_questions_flat})
            progress.set_description(f"{image_path.stem}")
            regions = self.extract_regions(image_path)
            for region_name, (aligned_image, weight) in regions.items():
                ticked = " - TICKED" if weight > weight_threshold else ""
                results[image_path.stem][region_name] = (
                    "X" if weight > weight_threshold else ""
                )
                cv2.imwrite(
                    (
                        extracted_folder
                        / f"{image_path.stem}-{region_name}{ticked}.png"
                    ).as_posix(),
                    aligned_image,
                )

        results = pd.DataFrame(results).T
        results.columns = pd.MultiIndex.from_tuples(self.get_all_question_names())
        results.to_excel((session_folder / "results.xlsx").as_posix())
        results.to_pickle((session_folder / "results.pkl").as_posix())
        return aligned_folder, results
