import shutil
from pathlib import Path

from pdf2image import convert_from_path
from tqdm import tqdm


def extract_from_pdf(src_file: Path, output_folder: Path, overwrite=False):
    output_folder = output_folder / src_file.stem
    if output_folder.exists() and not overwrite:
        return None

    raw_folder = output_folder / "01 - raw"
    shutil.rmtree(raw_folder, ignore_errors=True)
    raw_folder.mkdir(parents=True, exist_ok=True)

    pdf_to_images(src_file, raw_folder)

    return raw_folder


def pdf_to_images(pdf_path, output_folder):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path)

    # Save images to the specified output folder
    progress = tqdm(images)
    for i, image in enumerate(progress):
        page = i // 2
        side = "1-front" if i % 2 == 0 else "2-back"

        progress.set_description(f"Page {page+1:02d}")

        filename = f"{output_folder}/{page+1:02d}-{side}.png"
        image.save(filename, "PNG")
