from PIL import Image


def get_image(path: str):
	return Image.open(path)

