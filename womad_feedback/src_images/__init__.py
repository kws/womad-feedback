from pathlib import Path

IMAGE_PATH = Path(__file__).parent

IMAGES = list(IMAGE_PATH.glob("*.png"))
