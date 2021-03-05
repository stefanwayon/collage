from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from PIL import Image, ImageDraw
from random import randint

from collage.bric import fit_pictures_balanced


def make_pil_collage(images,
                     background_colour=(0, 0, 0, 0),
                     border_colour=(255, 255, 255, 255),
                     border=0,
                     spacing=0):
    image_sizes = [img.size for img in images]
    new_sizes, positions, canvas_size = fit_pictures_balanced(
        image_sizes, border=border, spacing=spacing, round_result=True)

    canvas = Image.new('RGBA', (ceil(canvas_size.w), ceil(canvas_size.h)),
                       background_colour)

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
parser.add_argument('--border', default=0, type=int)
parser.add_argument('--spacing', default=0, type=int)
parser.add_argument('image_file', nargs='+', type=Path)
args = parser.parse_args()

images = [Image.open(f) for f in args.image_file]


# images = [Image.new('RGB', (320, 220), tuple([randint(0, 150) for _ in range(3)]))
#           for _ in range(3)]
make_pil_collage(images, border=args.border, spacing=args.spacing).save('asdf.png')
