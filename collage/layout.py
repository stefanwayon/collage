from abc import abstractmethod
from dataclasses import dataclass
from collections import namedtuple
from functools import cached_property
from itertools import chain, islice
from numbers import Real
from typing import Iterable, Optional, List, Union, Tuple

LeafId = int
SizeOrTuple = Union['Size', Tuple[Real, Real]]


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
    def __add__(self, other: SizeOrTuple):
        other_w, other_h = other
        return Position(self.x + other_w, self.y + other_h)

    def __sub__(self, other: SizeOrTuple):
        other_w, other_h = other
        return Position(self.x - other_w, self.y - other_h)

    def __round__(self):
        return Size(*map(round, self))


_BoundingBoxChildren = Optional[List['BoundingBox']]


@dataclass(frozen=True)
class Constraint:
    is_height: bool
    positive_ids: List[LeafId]
    negative_ids: List[LeafId]
    result: Real = 0

    @classmethod
    def make_constraint(_,
                        is_height: bool,
                        positive_ids: List[LeafId],
                        negative_ids: List[LeafId],
                        result: Real = 0,
                        border: int = 0,
                        spacing: int = 0):
        result += (2 * len(negative_ids) * border
                   - 2 * len(positive_ids) * border)
        result += ((len(negative_ids) - 1) * spacing
                   - (len(positive_ids) - 1) * spacing)

        return Constraint(is_height, positive_ids, negative_ids, result)


class BoundingBox:
    is_horizontal: Optional[bool]
    children: _BoundingBoxChildren
    leaf_id: Optional[LeafId]
    width_ids: List[LeafId]
    height_ids: List[LeafId]

    def __init__(self, is_horizontal: Optional[bool] = None, *,
                 children: _BoundingBoxChildren = None,
                 leaf_id: Optional[LeafId] = None):
        if children is None and leaf_id is None:
            raise ValueError("Either children or leaf_id must be set")

        if children is not None and leaf_id is not None:
            raise ValueError("Only one of children or leaf_id must be set")

        if children is not None and is_horizontal is None:
            raise ValueError("Must set is_horizontal when there are children")

        object.__setattr__(self, 'is_horizontal', is_horizontal)
        object.__setattr__(self, 'children', children)
        object.__setattr__(self, 'leaf_id', leaf_id)

        if self.is_leaf:
            width_ids = [leaf_id]
            height_ids = [leaf_id]
        elif self.is_horizontal:
            # widths add up
            width_ids = sum([c.width_ids for c in children], [])

            # pick the child with fewest number of height components
            # to make constraints sparser
            height_ids = min([c.height_ids for c in children], key=len)
        else:
            # heights add up
            height_ids = sum([c.height_ids for c in children], [])

            # pick the child with fewest number of height components
            # to make constraints sparser
            width_ids = min([c.width_ids for c in children], key=len)

        object.__setattr__(self, 'width_ids', width_ids)
        object.__setattr__(self, 'height_ids', height_ids)

    @property
    def has_children(self) -> bool:
        return self.children is not None

    @property
    def is_leaf(self) -> bool:
        return self.leaf_id is not None

    @cached_property
    def n_images(self) -> int:
        if self.is_leaf:
            return 1
        else:
            assert self.children is not None
            return sum([c.n_images for c in self.children])

    def get_constraints(self, border: Real = 0, spacing: Real = 0) -> List[Constraint]:
        if self.is_leaf:
            constraints = []
        else:
            constraints = sum([c.get_constraints(border, spacing)
                               for c in self.children], [])

            if self.is_horizontal:
                # add a height constraints
                img_ids = [c.height_ids for c in self.children]
            else:
                # add a width constraints
                img_ids = [c.width_ids for c in self.children]

            for i in range(len(img_ids) - 1):
                constraints.append(
                    Constraint.make_constraint(
                        is_height=self.is_horizontal,
                        positive_ids=img_ids[i],
                        negative_ids=img_ids[i + 1],
                        border=border,
                        spacing=spacing))

        return constraints

    def compute_positions(self,
                          image_sizes: List[Size],
                          border: Real = 0,
                          spacing: Real = 0) -> Tuple[List[Position], Size]:
        def _recurse(bbox, top_left: Position = Position(0, 0)):
            if bbox.is_leaf:
                positions = [(bbox.leaf_id, top_left + (border, border))]
                size = image_sizes[bbox.leaf_id] + (2 * border, 2 * border)
            else:
                positions = []

                # add all the spacing to the size
                if bbox.is_horizontal:
                    size = Size(spacing * (len(bbox.children) - 1), 0)
                else:
                    size = Size(0, spacing * (len(bbox.children) - 1))

                for bbox_child in bbox.children:
                    # the last child shouldn’t have any spacing added, but the
                    # top corner is only passed amongst siblings and not up the
                    # tree, so the extra spacing will be discarded anyway
                    c_positions, c_size = _recurse(bbox_child, top_left)
                    positions.extend(c_positions)

                    if bbox.is_horizontal:
                        top_left = top_left + (c_size.w + spacing, 0)
                        size = c_size + (size.w, 0)
                    else:
                        top_left = top_left + (0, c_size.h + spacing)
                        size = c_size + (0, size.h)

            return positions, size

        positions, canvas_size = _recurse(self)
        assert len(positions) == len(image_sizes)

        # sort positions so they match the image size indices
        positions = [p[1] for p in sorted(positions)]

        return positions, canvas_size


class Layout(object):

    def __init__(
        self,
        n_images: int,
        width: Optional[Real] = None,
        height: Optional[Real] = None,
        border: Real = 0,
        spacing: Real = 0,
    ):
        self.n_images = n_images
        self.width = width
        self.height = height
        self.border = border
        self.spacing = spacing

    @abstractmethod
    def get_constraints(self, image_sizes: List[Size]) -> List[Constraint]:
        pass

    @abstractmethod
    def compute_positions(self, image_sizes: List[Size]) -> Tuple[List[Position], Size]:
        pass


class BoundingBoxLayout(Layout):
    def __init__(
        self,
        root_bbox: BoundingBox,
        width: Optional[Real] = None,
        height: Optional[Real] = None,
        border: Real = 0,
        spacing: Real = 0,
    ):
        super().__init__(
            root_bbox.n_images,
            width=width,
            height=height,
            border=border,
            spacing=spacing,
        )
        self.root_bbox = root_bbox

    def get_constraints(self, image_sizes: List[Size]) -> List[Constraint]:
        constraints = self.root_bbox.get_constraints(
            border=self.border, spacing=self.spacing
        )

        # for N images, the layout will yield N-1 constraints, so we add an
        # extra one to obtain a unique solution
        assert self.n_images - 1 == len(constraints)

        constraints.append(self._get_scale_constraint(image_sizes))

        return constraints

    def compute_positions(self, image_sizes: List[Size]) -> Tuple[List[Position], Size]:
        return self.root_bbox.compute_positions(image_sizes, self.border, self.spacing)

    def _get_scale_constraint(self, image_sizes: List[Size]) -> Constraint:
        if self.height is not None:
            border_height = 2 * len(self.root_bbox.height_ids) * self.border
            spacing_height = (len(self.root_bbox.height_ids) - 1) * self.spacing
            img_height = self.height - border_height - spacing_height

            # All image hights that are stacked vertically must add up to the
            # canvas height without borders and spacing
            return Constraint(
                is_height=True,
                positive_ids=self.root_bbox.height_ids,
                negative_ids=[],
                result=img_height,
            )
        else:
            border_width = 2 * len(self.root_bbox.width_ids) * self.border
            spacing_width = (len(self.root_bbox.width_ids) - 1) * self.spacing
            natural_width = sum([image_sizes[i].w for i in self.root_bbox.width_ids])
            img_width = (self.width or natural_width) - border_width - spacing_width

            # All image widths that are stacked horizontally must add up to the
            # canvas width without borders and spacing
            return Constraint(
                is_height=False,
                positive_ids=self.root_bbox.width_ids,
                negative_ids=[],
                result=img_width,
            )


class BalancedLayout(BoundingBoxLayout):
    def __init__(self,
                 n_images: int, *,
                 horizontal_root: bool = True,
                 width: Optional[Real] = None,
                 height: Optional[Real] = None,
                 border: Real = 0,
                 spacing: Real = 0):

        def _build_bbox_tree(start_id, end_id, is_horizontal):
            if end_id - start_id == 1:
                return BoundingBox(is_horizontal, leaf_id=start_id)
            else:
                mid_id = start_id + (end_id - start_id) // 2
                bbox_a = _build_bbox_tree(start_id, mid_id, not is_horizontal)
                bbox_b = _build_bbox_tree(mid_id, end_id, not is_horizontal)
                return BoundingBox(is_horizontal, children=[bbox_a, bbox_b])

        root_bbox = _build_bbox_tree(0, n_images, horizontal_root)
        super().__init__(root_bbox, width, height, border, spacing)


class GridLayout(BoundingBoxLayout):
    def __init__(self, n_images: int, *,
                 cols: Optional[int] = None,
                 rows: Optional[int] = None,
                 width: Optional[Real] = None,
                 height: Optional[Real] = None,
                 border: Real = 0,
                 spacing: Real = 0):

        if cols is None and rows is None:
            raise ValueError("Either cols or rows must be set")

        if cols is not None and rows is not None:
            raise ValueError("Only one of cols or rows must be set")

        num_groups = cols if cols is not None else rows
        assert num_groups is not None
        num_per_group = n_images / num_groups
        root_is_horizontal = rows is None

        bboxes = []
        for image_ids in _group_iter(range(n_images), num_per_group):
            leaf_bboxes = [BoundingBox(leaf_id=i) for i in image_ids]
            bbox = BoundingBox(not root_is_horizontal, children=leaf_bboxes)
            bboxes.append(bbox)

        root_bbox = BoundingBox(root_is_horizontal, children=bboxes)
        super().__init__(root_bbox, width, height, border, spacing)


def _group_iter(seq: Iterable, group_size: int | float):
    assert group_size > 0
    it = iter(seq)

    n_items = group_size
    items = []

    for item in it:
        n_items -= 1
        items.append(item)
        if n_items <= 0:
            yield items
            items = []
            n_items += group_size

    if len(items) > 0:
        yield items

    # while True:
    #     try:
    #         yield chain((next(it),), islice(it, group_size-1))
    #     except StopIteration:
    #         break
