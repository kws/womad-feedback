from collections import Counter
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from tqdm import tqdm

from womad_feedback.align import ImageAligner
from womad_feedback.average import create_average_image
from womad_feedback.average import least_black as _least_black
from womad_feedback.extract import extract_from_pdf, pdf_to_images
from womad_feedback.regions import RegionExtractor
from womad_feedback.src_images import IMAGES
from womad_feedback.textractor import extract_session

load_dotenv()

OUTPUT_FOLDER = Path("output")


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


@cli.command()
@click.option(
    "--out",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=OUTPUT_FOLDER,
)
def least_black(out: Path):
    _least_black(out)


@cli.command()
@click.option("--force", "-f", is_flag=True, default=False)
@click.argument(
    "filenames", nargs=-1, type=click.Path(exists=True, readable=True, path_type=Path)
)
def extract_regions(filenames, force):
    extractor = RegionExtractor()

    for session_name in filenames:
        click.echo(f"Extracting {session_name}")

        try:
            folder, results = extractor.extract_session(session_name, overwrite=force)
            if folder is None:
                click.echo(f"Skipping {session_name}")
            print(results)
        except Exception as e:
            click.echo(f"Error extracting {session_name}: {e}")


@cli.command()
@click.argument(
    "filenames", nargs=-1, type=click.Path(exists=True, readable=True, path_type=Path)
)
def extract_text(filenames):
    for session_name in filenames:
        click.echo(f"Extracting {session_name}")
        extract_session(session_name)


if __name__ == "__main__":
    cli()
