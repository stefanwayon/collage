from dataclasses import dataclass
from itertools import chain, islice
from numbers import Real
from typing import Iterable, Optional, List


_BoundingBoxChildren = Optional[List['BoundingBox']]


@dataclass(init=False, frozen=True)
class BoundingBox:
    is_horizontal: Optional[bool]
    children: _BoundingBoxChildren
    leaf_id: Optional[int]

    def __init__(self, is_horizontal: Optional[bool] = None, *,
                 children: _BoundingBoxChildren = None,
                 leaf_id: Optional[int] = None) -> None:
        if children is None and leaf_id is None:
            raise ValueError("Either children or leaf_id must be set")

        if children is not None and leaf_id is not None:
            raise ValueError("Only one of children or leaf_id must be set")

        if children is not None and is_horizontal is None:
            raise ValueError("Must set is_horizontal when there are children")

        object.__setattr__(self, 'is_horizontal', is_horizontal)
        object.__setattr__(self, 'children', children)
        object.__setattr__(self, 'leaf_id', leaf_id)

    @property
    def has_children(self):
        return self.children is not None

    @property
    def is_leaf(self):
        return self.leaf_id is not None


@dataclass(frozen=True)
class Layout:
    n_images: int
    root_bounding_box: BoundingBox
    width: Optional[Real] = None
    height: Optional[Real] = None
    spacing: Optional[Real] = None

    @classmethod
    def balanced_layout(_, n_images, horizontal_root=True) -> 'Layout':
        def _build_bbox_tree(start_id, end_id, is_horizontal):
            if end_id - start_id == 1:
                return BoundingBox(is_horizontal, leaf_id=start_id)
            else:
                mid_id = start_id + (end_id - start_id) // 2
                bbox_a = _build_bbox_tree(start_id, mid_id, not is_horizontal)
                bbox_b = _build_bbox_tree(mid_id, end_id, not is_horizontal)
                return BoundingBox(is_horizontal, children=[bbox_a, bbox_b])

        root_bbox = _build_bbox_tree(0, n_images, horizontal_root)
        return Layout(n_images, root_bbox)

    @classmethod
    def grid_layout(_, n_images: int, *,
                    cols: Optional[int] = None,
                    rows: Optional[int] = None) -> 'Layout':

        if cols is None and rows is None:
            raise ValueError("Either cols or rows must be set")

        if cols is not None and rows is not None:
            raise ValueError("Only one of cols or rows must be set")

        num_per_group = cols if cols is not None else rows
        root_is_horizontal = rows is not None

        bboxes = []
        for image_ids in _group_iter(range(n_images), num_per_group):
            leaf_bboxes = [BoundingBox(leaf_id=i) for i in image_ids]
            bbox = BoundingBox(not root_is_horizontal, children=leaf_bboxes)
            bboxes.append(bbox)

        root_bbox = BoundingBox(root_is_horizontal, children=bboxes)

        return Layout(n_images, root_bbox)


def _group_iter(seq: Iterable, group_size: int):
    assert group_size > 0
    it = iter(seq)
    while True:
        try:
            yield chain((next(it),), islice(it, group_size-1))
        except StopIteration:
            break
