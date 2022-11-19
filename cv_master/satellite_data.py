import os
import pandas as pd
from typing import List, Tuple, Set, Generator

from .dataclass import Patch, Coordinate, Grid2DKernel
from .grid import Grid2D

class SatelliteImageData:
    START_COORD = Coordinate(65, 2)
    END_COORD = Coordinate(369, 383)
    BASE_DIR = os.path.join(os.path.split(__file__)[0], 'data/')
    COLUMNES = ['y', 'x', 'label', 'ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']

    def __init__(self, kernel: Grid2DKernel, images: List[str] = None):
        """
        Parameters
        ----------
        kernel: The Grid Kernel for splitting purposes
        images: The list of images names to read in, default to all 3 images
        """
        # the overlaying grid with kernel and default starting and ending coordinates for images
        self.grid = Grid2D(SatelliteImageData.START_COORD, SatelliteImageData.END_COORD, kernel)
        self.df = self._read_data(images)
        self.images = images
    
    def _read_data(self, images: List[str] = None):
        base_dir = SatelliteImageData.BASE_DIR
        images_names = images if images is not None else os.listdir(base_dir)

        data = []
        for i, file_name in enumerate(images_names):
            chunk = pd.read_csv(os.path.join(base_dir, file_name), delim_whitespace=True, header=None)
            chunk.columns = SatelliteImageData.COLUMNES
            chunk['image'] = i
            data.append(chunk)

        data = pd.concat(data)
        data.sort_values(['x', 'y'], inplace=True)
        return data

    def get_data_patch(self, patch: Patch) -> pd.DataFrame:
        """Map the image data to the corresponding patch coordinates"""
        df = self.df

        x_lb, x_ub = patch.x_bound
        y_lb, y_ub = patch.y_bound
        
        x_bound = (df.x >= x_lb) & (df.x <= x_ub)
        y_bound = (df.y >= y_lb) & (df.y <= y_ub)

        return df[x_bound & y_bound]
    
    def get_data_patches(self, patches: Set[Patch]) -> pd.DataFrame:
        data = []
        for patch in patches:
            data.append(self.get_data_patch(patch))
        return pd.concat(data)
    
    def iter_train_validate(self) -> Generator[Tuple[Set[Patch], Patch], None, None]:
        """
        Nested Train Validate Splitting
        - No test nesting in this procedure
        - Validate patch starts from bottom left
        - The rest are the training set except the neighbors around validate patches
        """
        for i, j, patch in self.grid.iter_patches():
            validate = patch
            neighbors = self.grid.get_neighbors(i, j)
            train = self.grid.patch_set.difference(neighbors.union({validate}))

            yield train, validate
    
    def iter_train_validate_test(self) -> Generator[Tuple[Set[Patch], Patch, Patch], None, None]:
        """
        Nested Train Validate Test Splitting
        - Validate patch starts from bottom left
        - Test patch starts from top right
        - The rest are the training set except the neighbors around validate and test patches
        """
        val_patches = list(self.grid.iter_patches())
        test_patches = reversed(val_patches)

        for (vi, vj, vpatch), (ti, tj, tpatch) in zip(val_patches, test_patches):
            validate = vpatch
            test = tpatch
            neighbors = self.grid.get_neighbors(vi, vj).union(self.grid.get_neighbors(ti, tj))

            # ignore the cases when validate and test overlaps
            if test in neighbors or validate in neighbors:
                continue

            train = self.grid.patch_set.difference(neighbors.union({validate, test}))

            yield train, validate, test
