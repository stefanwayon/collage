import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from numbers import Real
from typing import List, Tuple, Union

from .layout import Layout

SizeOrTuple = Tuple[Real, Real]


class Size(namedtuple('Size', ['w', 'h'])):
    def __add__(self, other: SizeOrTuple):
        other_w, other_h = other
        return Size(self.w + other_w, self.h + other_h)

    def __sub__(self, other: SizeOrTuple):
        other_w, other_h = other
        return Size(self.w - other_w, self.h - other_h)

    def __round__(self):
        return Size(*map(round, self))

    def __truediv__(self, divisor: Union[Real, SizeOrTuple]):
        if isinstance(divisor, Real):
            return Size(self.w / divisor, self.h / divisor)
        else:
            div_w, div_h = divisor
            return Size(self.w / div_w, self.h / div_h)


class Position(namedtuple('Position', ['x', 'y'])):
    def __add__(self, other: Union[Tuple[Real, Real], Size]):
        other_w, other_h = other
        return Position(self.x + other_w, self.y + other_h)

    def __round__(self):
        return Size(*map(round, self))


@dataclass(frozen=True)
class Constraint:
    is_height: bool
    positive_ids: list
    negative_ids: list
    result: Real = 0


def fit_pictures_balanced(image_sizes, layout='horizontal', round_result=False):
    image_sizes = [Size(h, w) for h, w in image_sizes]
    layout = Layout.balanced_layout(len(image_sizes), layout == 'horizontal')

    return fit_pictures(image_sizes, layout, round_result)


def fit_pictures_grid(image_sizes, *, cols=None, rows=None, round_result=False):
    image_sizes = [Size(h, w) for h, w in image_sizes]
    layout = Layout.grid_layout(len(image_sizes), cols=cols, rows=rows)

    return fit_pictures(image_sizes, layout, round_result)


def fit_pictures(image_sizes: List[Size], layout: Layout, round_result=False):
    if round_result:
        new_sizes = _compute_integer_image_sizes(image_sizes, layout)
    else:
        new_sizes = _compute_rescaled_image_sizes(image_sizes, layout)

    positions, canvas_size = _compute_positions(new_sizes, layout)

    return new_sizes, positions, canvas_size


def _compute_rescaled_image_sizes(image_sizes: List[Size], layout: Layout):
    constraints = _get_constraints(image_sizes, layout)

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


def _compute_integer_image_sizes(image_sizes: List[Size], layout: Layout):
    import mip
    constraints = _get_constraints(image_sizes, layout)
    aspect_ratios = [h/w for w, h in image_sizes]

    # set up a mixed-integer program, and solve it
    n_images = len(image_sizes)

    m = mip.Model()
    var_widths = [m.add_var(var_type=mip.INTEGER) for _ in range(n_images)]
    var_heights = [m.add_var(var_type=mip.INTEGER) for _ in range(n_images)]

    for c in constraints:
        if c.is_height:
            vars = ([var_heights[i] for i in c.positive_ids] +
                    [-var_heights[i] for i in c.negative_ids])
        else:
            vars = ([var_widths[i] for i in c.positive_ids] +
                    [-var_widths[i] for i in c.negative_ids])

        m.add_constr(mip.xsum(vars) == c.result)

    # the errors come from a deviation in aspect ratio
    var_errs = [m.add_var(var_type=mip.CONTINUOUS) for _ in range(n_images)]
    for err, w, h, ar in zip(var_errs, var_widths, var_heights, aspect_ratios):
        m.add_constr(err == h - w * ar)

    # To minimise error, we need to create a convex cost function. Common
    # options are either abs(err) or err ** 2. However, both these functions are
    # non-linear, so cannot be directly computed in MIP. We can represent abs
    # exactly with a type-1 SOS, and approximate ** 2 with a type-2 SOS. Here we
    # use abs.

    var_errs_pos = [m.add_var(var_type=mip.CONTINUOUS) for _ in range(n_images)]
    var_errs_neg = [m.add_var(var_type=mip.CONTINUOUS) for _ in range(n_images)]
    var_abs_errs = [m.add_var(var_type=mip.CONTINUOUS) for _ in range(n_images)]

    for abs_err, err, err_pos, err_neg in zip(var_abs_errs,
                                              var_errs,
                                              var_errs_pos,
                                              var_errs_neg):
        # err_pos and err_neg are both positive representing each side of the
        # abs function. Only one will be non-zero (SOS Type-1).
        m.add_constr(err == err_pos - err_neg)
        m.add_constr(abs_err == err_pos + err_neg)
        m.add_sos([(err_pos, 1), (err_neg, -1)], sos_type=1)

    m.objective = mip.minimize(mip.xsum(var_abs_errs))
    m.optimize(max_seconds=30)

    new_sizes = [Size(int(w.x), int(h.x))
                 for w, h in zip(var_widths, var_heights)]
    return new_sizes


def _get_constraints(image_sizes, layout):
    constraints = _get_bbox_constraints(layout.root_bounding_box)

    # for N images, the layout will yield N-1 constraints, so we add an extra
    # one to obtain a unique solution
    assert layout.n_images - 1 == len(constraints)

    total_width = (layout.width
                   or sum([image_sizes[i].w
                           for i in layout.root_bounding_box.width_ids]))
    constraints.append(
        Constraint(is_height=False,
                   positive_ids=layout.root_bounding_box.width_ids,
                   negative_ids=[],
                   result=total_width)
    )

    return constraints


def _get_bbox_constraints(bbox):
    if bbox.is_leaf:
        constraints = []
    else:
        constraints = sum([_get_bbox_constraints(c) for c in bbox.children], [])

        if bbox.is_horizontal:
            c_heights = [c.height_ids for c in bbox.children]

            # add a height constraints
            for i in range(len(c_heights) - 1):
                constraints.append(
                    Constraint(is_height=True,
                               positive_ids=c_heights[i],
                               negative_ids=c_heights[i + 1]))
        else:
            c_widths = [c.width_ids for c in bbox.children]

            # add a width constraints
            for i in range(len(c_widths) - 1):
                constraints.append(
                    Constraint(is_height=False,
                               positive_ids=c_widths[i],
                               negative_ids=c_widths[i + 1]))

    return constraints


def _compute_positions(image_sizes: List[Size], layout: Layout):
    def _recurse(bbox, top_left: Position = Position(0, 0)):
        if bbox.is_leaf == 1:
            positions = [(bbox.leaf_id, top_left)]
            size = image_sizes[bbox.leaf_id]
        else:
            positions = []
            size = Size(0, 0)
            for bbox_child in bbox.children:
                c_positions, c_size = _recurse(bbox_child, top_left)
                positions.extend(c_positions)

                if bbox.is_horizontal:
                    top_left = top_left + (c_size.w, 0)
                    size = c_size + (size.w, 0)
                else:
                    top_left = top_left + (0, c_size.h)
                    size = c_size + (0, size.h)

        return positions, size

    positions, canvas_size = _recurse(layout.root_bounding_box)
    assert len(positions) == len(image_sizes)

    # sort positions so they match the image size indices
    positions = [p[1] for p in sorted(positions)]

    return positions, canvas_size
