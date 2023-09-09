import json
from pathlib import Path

import boto3
import yaml
from tqdm import tqdm


def get_image_text(image: Path) -> dict:
    textract_client = boto3.client("textract", region_name="us-east-1")
    response = textract_client.analyze_document(
        Document={
            "Bytes": image.read_bytes(),
        },
        FeatureTypes=["FORMS"],
    )
    return response


def extract_session(session_folder: Path):
    raw_dir = session_folder / "01 - raw"
    text_dir = session_folder / "04 - text"
    text_dir.mkdir(exist_ok=True)

    image_paths = list(raw_dir.glob("*-back.png"))
    image_paths.sort()

    progress = tqdm(image_paths)
    for image in progress:
        progress.set_description(image.name)
        text_path = text_dir / image.with_suffix(".yaml").name
        if text_path.exists():
            continue

        response = get_image_text(image)
        text_path.write_text(yaml.safe_dump(response))

        lines = []
        for block in response["Blocks"]:
            if block["BlockType"] == "LINE":
                if "Text" in block:
                    line = block["Text"]
                    only_chars_line = "".join([c for c in line if c.isalpha()])
                    if only_chars_line not in [
                        "ANYCOMMENTSORSUGGESTIONS",
                        "THANKYOUFORYOURTIME",
                    ]:
                        lines.append(line)

        text_path.with_suffix(".txt").write_text("\n".join(lines))
