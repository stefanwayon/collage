import numpy as np

from typing import List, Tuple

from .layout import Layout, GridLayout, BalancedLayout, Size, Position


def fit_pictures_balanced(
    image_sizes,
    layout="horizontal",
    width=None,
    border=0,
    spacing=0,
    round_result=False,
) -> Tuple[List[Size], List[Position], Size]:
    if round_result:
        assert isinstance(border, int)
        assert isinstance(spacing, int)

    image_sizes = [Size(w, h) for w, h in image_sizes]
    layout = BalancedLayout(
        len(image_sizes),
        horizontal_root=(layout == "horizontal"),
        width=width,
        border=border,
        spacing=spacing,
    )

    return fit_pictures(image_sizes, layout, round_result)


def fit_pictures_grid(
    image_sizes,
    *,
    cols=None,
    rows=None,
    width=None,
    border=0,
    spacing=0,
    round_result=False
) -> Tuple[List[Size], List[Position], Size]:
    image_sizes = [Size(w, h) for w, h in image_sizes]
    layout = GridLayout(
        len(image_sizes),
        cols=cols,
        rows=rows,
        width=width,
        border=border,
        spacing=spacing,
    )

    return fit_pictures(image_sizes, layout, round_result)


def fit_pictures(image_sizes: List[Size], layout: Layout, round_result=False) -> Tuple[List[Size], List[Position], Size]:
    if round_result:
        new_sizes = _compute_integer_image_sizes(image_sizes, layout)
    else:
        new_sizes = _compute_rescaled_image_sizes(image_sizes, layout)

    positions, canvas_size = layout.compute_positions(new_sizes)

    return new_sizes, positions, canvas_size


def _compute_rescaled_image_sizes(image_sizes: List[Size], layout: Layout) -> List[Size]:
    constraints = layout.get_constraints(image_sizes)

    # set up and a linear equation system from the constraints, and solve it
    n_images = len(image_sizes)
    n_constraints = len(constraints)

    aspect_ratios = np.array([h/w for w, h in image_sizes])
    A = np.zeros((n_constraints, n_images))  # constraints matrix
    b = np.zeros(n_constraints)  # rhs of equation Aw = b

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


def _compute_integer_image_sizes(image_sizes: List[Size], layout: Layout) -> List[Size]:
    import mip
    constraints = layout.get_constraints(image_sizes)
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
