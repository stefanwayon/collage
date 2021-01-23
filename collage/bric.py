import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from numbers import Real
from typing import List

from .layout import Layout


Size = namedtuple('Size', ['w', 'h'])
Position = namedtuple('Position', ['x', 'y'])


@dataclass(frozen=True)
class Constraint:
    is_height: bool
    positive_ids: list
    negative_ids: list
    result: Real = 0


def fit_pictures_balanced(image_sizes, layout='horizontal'):
    image_sizes = [Size(h, w) for h, w in image_sizes]
    layout = Layout.balanced_layout(len(image_sizes), layout == 'horizontal')

    return fit_pictures(image_sizes, layout)


def fit_pictures_grid(image_sizes, *, cols=None, rows=None):
    image_sizes = [Size(h, w) for h, w in image_sizes]
    layout = Layout.grid_layout(len(image_sizes), cols=cols, rows=rows)

    return fit_pictures(image_sizes, layout)


def fit_pictures(image_sizes: List[Size], layout: Layout):
    new_sizes = _compute_rescaled_image_sizes(image_sizes, layout)
    positions, canvas_size = _compute_positions(new_sizes, layout)

    return new_sizes, positions, canvas_size


def _compute_rescaled_image_sizes(image_sizes: List[Size], layout: Layout):
    widths, _, constraints = _get_bbox_constraints(
        layout.root_bounding_box)

    # for N images, the layout will yield N-1 constraints, so we add an extra
    # one to obtain a unique solution
    assert layout.n_images - 1 == len(constraints)

    constraints.append(
        Constraint(is_height=False,
                   positive_ids=widths,
                   negative_ids=[],
                   result=sum([image_sizes[i].w for i in widths]))
    )

    # set up and a linear equation system from the constraints, and solve it
    n_images = len(image_sizes)

    aspect_ratios = np.array([h/w for w, h in image_sizes])
    A = np.zeros((n_images, n_images))  # constraints matrix
    b = np.zeros(n_images)              # rhs of equation Aw = b

    for c_idx, c in enumerate(constraints):
        for img_idx in c.positive_ids:
            A[c_idx, img_idx] = aspect_ratios[img_idx] if c.is_height else 1

        for img_idx in c.negative_ids:
            A[c_idx, img_idx] = -(aspect_ratios[img_idx] if c.is_height else 1)

        b[c_idx] = c.result

    new_widths = np.linalg.solve(A, b)
    new_heights = aspect_ratios * new_widths

    new_sizes = [Size(w, h) for w, h in zip(new_widths, new_heights)]

    return new_sizes


def _get_bbox_constraints(bbox):
    if bbox.is_leaf:
        widths = [bbox.leaf_id]
        heights = [bbox.leaf_id]
        constraints = []
    else:
        c_widths, c_heights, c_constraints = zip(*[_get_bbox_constraints(c)
                                                   for c in bbox.children])

        if bbox.is_horizontal:
            # widths add up
            widths = sum(c_widths, [])

            # pick the child with fewest number of height components
            # to make constraints sparser
            heights = min(c_heights, key=len)

            constraints = sum(c_constraints, [])

            # add a height constraints
            for i in range(len(c_heights) - 1):
                constraints.append(
                    Constraint(is_height=True,
                               positive_ids=c_heights[i],
                               negative_ids=c_heights[i + 1]))
        else:
            # heights add up
            heights = sum(c_heights, [])

            # pick the child with fewest number of width components
            # to make constraints sparser
            widths = min(c_widths, key=len)

            constraints = sum(c_constraints, [])

            # add a width constraints
            for i in range(len(c_widths) - 1):
                constraints.append(
                    Constraint(is_height=False,
                               positive_ids=c_widths[i],
                               negative_ids=c_widths[i + 1]))

    return widths, heights, constraints


def _compute_positions(image_sizes: List[Size], layout: Layout):
    def _c(bbox, top_left: Position = Position(0, 0)):
        if bbox.is_leaf == 1:
            positions = [(bbox.leaf_id, top_left)]
            size = image_sizes[bbox.leaf_id]
        else:
            positions = []
            size = Size(0, 0)
            for bbox_child in bbox.children:
                c_positions, c_size = _c(bbox_child, top_left)
                positions.extend(c_positions)

                if bbox.is_horizontal:
                    top_left = Position(top_left.x + c_size.w, top_left.y)
                    size = Size(size.w + c_size.w, c_size.h)
                else:
                    top_left = Position(top_left.x, top_left.y + c_size.h)
                    size = Size(c_size.w, size.h + c_size.h)

        return positions, size

    positions, canvas_size = _c(layout.root_bounding_box)
    assert len(positions) == len(image_sizes)

    # sort positions so they match the image size indices
    positions = [p[1] for p in sorted(positions)]

    return positions, canvas_size
