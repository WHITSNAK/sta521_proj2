from typing import List, Tuple, Set, Generator

from .dataclass import Coordinate, Grid2DKernel, Patch

class Grid2D:
    def __init__(self, start_coord: Coordinate, end_coord: Coordinate, kernel: Grid2DKernel):
        self.kernel = kernel
        self._patch = Patch(x0y0=start_coord, x1y1=end_coord)
        self._subgrids = self._get_nonoverlapping_subgrids()
        self.patch_set = {patch for row in self._subgrids for patch in row}
        self.shape = (len(self._subgrids), len(self._subgrids[0]))
    
    def __getitem__(self, index):
        return self._subgrids[index[0]][index[1]]

    def _get_nonoverlapping_subgrids(self) -> List[List[Patch]]:
        """Dividing the entire polygon down to non-overlapping mini grids by the kernel size"""
        x0, y0 = self._patch.x0y0
        x1, y1 = self._patch.x1y1
        dx, dy = self.kernel

        # x on columns, y on rows
        grid = []
        for y in range(y0, y1, dy):
            row = []
            for x in range(x0, x1, dx):
                row.append(Patch.from_one_point(Coordinate(x, y), self.kernel))
            grid.append(row)
        return grid
    
    def get_neighbors(self, i, j) -> Set[Patch]:
        """Returns a set of 8 neighboring patches that are within the valid boundary"""
        x_max, y_max = self.shape
        all_indices = [
            (i-1, j-1),
            (i-1, j),
            (i-1, j+1),
            (i, j-1),
            (i, j+1),
            (i+1, j-1),
            (i+1, j),
            (i+1, j+1),
        ]

        neighbors = set()
        for x, y in all_indices:
            is_valid_x = (x >= 0) & (x < x_max)
            is_valid_y = (y >= 0) & (y < y_max)

            # ignore those that is out of either side of boundaries
            if not (is_valid_x & is_valid_y): continue

            neighbors.add(self[x, y])
        
        return neighbors
    
    def iter_patches(self) -> Generator[Tuple[int, int, Patch], None, None]:
        """Yield patches from left to right along x-axis, row by row"""
        for i, row in enumerate(self._subgrids):
            for j, patch in enumerate(row):
                yield i, j, patch
