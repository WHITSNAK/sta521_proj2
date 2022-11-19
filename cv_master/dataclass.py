from __future__ import annotations

import matplotlib.patches as patches
from typing import NamedTuple

class Coordinate(NamedTuple):
    x: int
    y: int

class Grid2DKernel(NamedTuple):
    dx: int
    dy: int

class Patch(NamedTuple):
    x0y0: Coordinate
    x1y1: Coordinate

    @property
    def x_bound(self):
        return (self.x0y0.x, self.x1y1.x)
    
    @property
    def y_bound(self):
        return (self.x0y0.y, self.x1y1.y)

    @staticmethod
    def from_one_point(start_coord: Coordinate, kernel: Grid2DKernel) -> Patch:
        x0, y0 = start_coord
        dx, dy = kernel

        return Patch(Coordinate(int(x0), int(y0)), Coordinate(int(x0+dx-1), int(y0+dy-1)))
    
    def to_matplotlib(self, **kws):
        width = self.x1y1.x - self.x0y0.x
        height = self.x1y1.y - self.x0y0.y
        return patches.Rectangle(self.x0y0, width, height, **kws)
