from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from PIL import Image, ImageDraw

from collage.bric import fit_pictures_balanced


def make_pil_collage(
    images,
    background_colour=(0, 0, 0, 255),
    border_colour=(255, 255, 255, 255),
    width=None,
    border=0,
    spacing=0,
):
    image_sizes = [img.size for img in images]
    new_sizes, positions, canvas_size = fit_pictures_balanced(
        image_sizes, width=width, border=border, spacing=spacing, round_result=True
    )

    canvas = Image.new(
        "RGB", (ceil(canvas_size.w), ceil(canvas_size.h)), background_colour
    )

    if border > 0:
        draw = ImageDraw.Draw(canvas)

    for img, size, position in zip(images, new_sizes, positions):
        if border > 0:
            top_left = position - (border, border)
            bottom_right = position + size + (border - 1, border - 1)
            draw.rectangle([top_left, bottom_right], fill=border_colour)

        img = img.resize((round(size.w), round(size.h)), Image.BICUBIC)
        canvas.paste(img, (round(position.x), round(position.y)))

    return canvas


parser = ArgumentParser('collage')
parser.add_argument(
    "--border",
    default=0,
    type=int,
    help="Width of the border around each image (in pixels). "
    "The border is drawn around each image individually.",
)
parser.add_argument(
    "--spacing",
    default=0,
    type=int,
    help="Spacing between images (in pixels). "
    "The spacing is the gap between adjacent images.",
)
parser.add_argument(
    "--output_width",
    default=None,
    type=int,
    help="Width of the output image (in pixels). "
    "If not specified, the output width is determined automatically.",
)
parser.add_argument(
    "--output", default="output.png", type=str, help="Output image file name."
)
parser.add_argument(
    "image_file", nargs="+", type=Path, help="Paths to the input image files."
)
args = parser.parse_args()

images = [Image.open(f) for f in args.image_file]

make_pil_collage(
    images, width=args.output_width, border=args.border, spacing=args.spacing
).save(args.output)
