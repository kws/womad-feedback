import shutil
from collections import Counter
from pathlib import Path

import click
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from tqdm import tqdm

from womad_feedback.align import ImageAligner
from womad_feedback.average import create_average_image
from womad_feedback.extract import extract_from_pdf, pdf_to_images
from womad_feedback.src_images import IMAGES

OUTPUT_FOLDER = Path("output")


COORDINATES_2022 = [
    (578, 422, 75, 75, 0, 1750),
    (815, 422, 75, 75, 100, 1750),
    (680, 530, 75, 75, 0, 1825),
    (965, 530, 75, 75, 100, 1825),
    (818, 656, 75, 75, 0, 1900),
    (1274, 656, 75, 75, 100, 1900),
    (460, 768, 75, 75, 0, 1975),
    (723, 768, 75, 75, 100, 1975),
    (968, 768, 75, 75, 200, 1975),
    (1295, 768, 75, 75, 300, 1975),
    (1137, 881, 75, 75, 0, 2050),
    (1320, 881, 75, 75, 100, 2050),
    (1514, 881, 75, 75, 200, 2050),
    (1044, 977, 75, 75, 0, 2125),
    (1247, 977, 75, 75, 100, 2125),
]

COORDINATES_2023 = [
    (780, 294, 75, 75, 0, 1750),
    (1052, 294, 75, 75, 100, 1750),
    (1241, 294, 75, 75, 200, 1750),
    (626, 389, 75, 75, 0, 1825),
    (990, 389, 75, 75, 100, 1825),
    (1418, 389, 75, 75, 200, 1825),
    (908, 488, 75, 75, 0, 1900),
    (1301, 480, 75, 75, 100, 1900),
    (380, 720, 640, 50, 0, 1975),
    (380, 845, 640, 50, 0, 2025),
    (380, 960, 640, 50, 0, 2075),
    (380, 1060, 640, 50, 0, 2125),
]


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--out",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=OUTPUT_FOLDER,
)
@click.option("--force", "-f", is_flag=True, default=False)
@click.argument(
    "filenames", nargs=-1, type=click.Path(exists=True, readable=True, path_type=Path)
)
def extract(filenames, out, force):
    for session_name in filenames:
        click.echo(f"Extracting {session_name}")
        try:
            folder = extract_from_pdf(session_name, out, overwrite=force)
            if folder is None:
                click.echo(f"Skipping {session_name}")
        except Exception as e:
            click.echo(f"Error extracting {session_name}: {e}")


@cli.command()
@click.option("--force", "-f", is_flag=True, default=False)
@click.argument(
    "filenames", nargs=-1, type=click.Path(exists=True, readable=True, path_type=Path)
)
def align(filenames, force):
    aligner = ImageAligner()
    for file_name in IMAGES:
        image = cv2.imread(file_name.as_posix(), 0)
        aligner.add_base_image(file_name.stem, image)

    for session_name in filenames:
        click.echo(f"Extracting {session_name}")

        try:
            folder = aligner.align_session(session_name, overwrite=force)
            if folder is None:
                click.echo(f"Skipping {session_name}")
        except Exception as e:
            click.echo(f"Error extracting {session_name}: {e}")


@cli.command()
@click.option(
    "--out",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=OUTPUT_FOLDER,
)
def average(out):
    create_average_image(out)


def something_else():
    base_name = Path(filename).stem
    output_folder = OUTPUT_FOLDER / base_name
    shutil.rmtree(output_folder, ignore_errors=True)

    raw_folder = output_folder / "raw"
    final_folder = output_folder / "final"
    aligned_folder = output_folder / "aligned"

    raw_folder.mkdir(parents=True, exist_ok=True)
    final_folder.mkdir(parents=True, exist_ok=True)
    aligned_folder.mkdir(parents=True, exist_ok=True)

    pdf_to_images(filename, raw_folder)

    front_images = list(raw_folder.glob("*-front.png"))

    aligner = ImageAligner()
    for file_name in IMAGES:
        image = cv2.imread(file_name.as_posix(), 0)
        aligner.add_base_image(file_name.stem, image)

    max_h = max_w = 0
    for file_name in tqdm(front_images):
        image = cv2.imread(file_name.as_posix(), 0)
        match_name, _ = aligner.find_best_match(image)
        aligned_image = aligner.align(image, match_name)

        stem = file_name.stem
        extension = file_name.suffix
        aligned_name = f"{stem}-{match_name}.{extension}"

        max_h = max(max_h, aligned_image.shape[0])
        max_w = max(max_w, aligned_image.shape[1])

        cv2.imwrite((aligned_folder / aligned_name).as_posix(), aligned_image)

    # Create a new PDF file
    c = canvas.Canvas(
        (OUTPUT_FOLDER / f"{base_name}.pdf").as_posix(),
        pagesize=(max_w, max_h * 2 + 450),
    )

    cntr = Counter()
    for file_name in tqdm(sorted(aligned_folder.glob("*.png"))):
        page, side, year = file_name.stem.split("-")
        year = int(year[:4])
        back_name = f"{page}-back.png"
        cntr[year] += 1
        sequence = cntr[year]

        _, page = page.split("_")
        page = int(page) + 1

        front_image = cv2.imread(file_name.as_posix(), 0)
        back_image = cv2.imread((raw_folder / back_name).as_posix(), 0)

        back_image = resize_image_to(front_image, back_image)

        padding = np.ones((450, front_image.shape[1]), dtype=np.uint8) * 255
        stacked_img = np.vstack((padding, front_image, back_image))

        COORDINATES = COORDINATES_2023 if year == 2023 else COORDINATES_2022
        THRESHOLD_VALUE = 3700

        for x, y, w, h, dest_x, dest_y in COORDINATES:
            # dest_y += 550
            y += 450
            dest_y -= 1750
            roi = stacked_img[y : y + h, x : x + w]
            stacked_img[dest_y : dest_y + h, dest_x : dest_x + w] = roi

            if w == 75:
                _, thresholded_roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
                non_zero_count = cv2.countNonZero(thresholded_roi)
                if non_zero_count < THRESHOLD_VALUE:
                    pass
                    cv2.putText(
                        stacked_img,
                        "X",
                        (dest_x + 30, dest_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                # cv2.putText(stacked_img, f'{non_zero_count}', (dest_x+30, dest_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(
            stacked_img,
            f"{base_name} - {page}",
            (500, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        outname = f"{year}-{sequence:02d}.png"
        out_path = (final_folder / outname).as_posix()
        cv2.imwrite(out_path, stacked_img)

        c.drawImage(
            out_path, 0, 0, width=stacked_img.shape[1], height=stacked_img.shape[0]
        )
        c.showPage()

    c.save()


def resize_image_to(source_image, target_image):
    height1, width1 = source_image.shape
    height2, width2 = target_image.shape

    scale_factor = width1 / width2
    new_height = int(height2 * scale_factor)
    target_image = cv2.resize(target_image, (width1, new_height))

    if new_height > height1:
        target_image = target_image[:height1, :, :]
    elif new_height < height1:
        padding = np.zeros((height1 - new_height, width1), dtype=np.uint8)
        target_image = np.vstack((target_image, padding))

    return target_image


if __name__ == "__main__":
    cli()
