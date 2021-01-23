import numpy as np

from collections import namedtuple

Size = namedtuple('Size', ['h', 'w'])
Position = namedtuple('Position', ['x', 'y'])


def fit_pictures(image_sizes, layout='horizontal', spacing=0):
    image_sizes = [Size(h, w) for h, w in image_sizes]
    return _compute_layout(image_sizes, layout == 'horizontal', spacing)


def _compute_layout(image_sizes, is_horizontal, spacing):
    n_images = len(image_sizes)

    aspect_ratios = np.array([h/w for h, w in image_sizes])
    constraints = np.zeros((n_images, n_images))

    widths, heights, constraints_w, constraints_h = _get_bric_constraints(
        list(range(n_images)), is_horizontal)

    cn = 0
    for cw in constraints_w:
        for idx in cw:
            if idx > 0:
                constraints[cn, idx - 1] = 1
            else:
                constraints[cn, -idx - 1] = -1
        cn += 1

    for ch in constraints_h:
        for idx in ch:
            if idx > 0:
                constraints[cn, idx - 1] = aspect_ratios[idx - 1]
            else:
                constraints[cn, -idx - 1] = -aspect_ratios[-idx - 1]
        cn += 1

    total_width = 0
    for idx in widths:
        constraints[-1, idx - 1] = 1
        total_width += image_sizes[idx - 1].w

    b = np.zeros(n_images)
    b[-1] = total_width

    print(constraints)
    print(b)

    new_widths = np.linalg.solve(constraints, b)
    new_heights = aspect_ratios * new_widths

    new_sizes = [Size(h, w) for h, w in zip(new_heights, new_widths)]
    positions = _compute_positions(new_sizes, is_horizontal, spacing)

    canvas_height = sum([new_sizes[idx - 1].h for idx in heights])
    canvas_height += (len(heights) - 1) * spacing
    canvas_width = sum([new_sizes[idx - 1].w for idx in widths])
    canvas_width += (len(widths) - 1) * spacing
    canvas_size = Size(canvas_height, canvas_width)

    return canvas_size, new_sizes, positions


def _get_bric_constraints(images, is_horizontal):
    n_images = len(images)

    if n_images == 1:
        # index from 1 to allow for sign to indicate sign of constraint
        widths = [images[0] + 1]
        heights = [images[0] + 1]
        constraints_w = []
        constraints_h = []

    else:
        w1, h1, cw1, ch1 = _get_bric_constraints(images[:n_images // 2],
                                                 not is_horizontal)
        w2, h2, cw2, ch2 = _get_bric_constraints(images[n_images // 2:],
                                                 not is_horizontal)

        if is_horizontal:
            # widths add up, heights yield constraints
            widths = w1 + w2
            # keep the shortest height to make linear system sparser
            heights = h1 if len(h1) < len(h2) else h2
            constraints_w = cw1 + cw2
            constraints_h = ch1 + ch2 + [h1 + [-h for h in h2]]
        else:
            # heights add up, widths yield constraints
            heights = h1 + h2
            # keep the shortest width to make linear system sparser
            widths = w1 if len(w1) < len(w2) else w2
            constraints_h = ch1 + ch2
            constraints_w = cw1 + cw2 + [w1 + [-w for w in w2]]

    return widths, heights, constraints_w, constraints_h


def _compute_positions(image_sizes, is_horizontal, spacing):
    positions = [None] * len(image_sizes)

    def _c(images, is_horizontal, top_left):
        n_images = len(images)

        if n_images == 1:
            positions[images[0]] = top_left
            return image_sizes[images[0]]
        else:
            size1 = _c(images[:n_images // 2], not is_horizontal, top_left)

            if is_horizontal:
                top_left = Position(top_left.x + size1.w + spacing, top_left.y)
            else:
                top_left = Position(top_left.x, top_left.y + size1.h + spacing)

            size2 = _c(images[n_images // 2:], not is_horizontal, top_left)

            if is_horizontal:
                return Size(size1.h, size1.w + size2.w + spacing)
            else:
                return Size(size1.h + size2.h + spacing, size1.w)

    _c(list(range(len(image_sizes))), is_horizontal, Position(0, 0))

    return positions
