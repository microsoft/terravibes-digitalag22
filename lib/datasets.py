from typing import Any, Callable, Dict, Optional, Sequence

import rasterio
import torch
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from torchgeo.datasets import CDL as tgCDL
from torchgeo.datasets import BoundingBox, RasterDataset


class NDVIDataset(RasterDataset):
    filename_glob = "ndvi_stack_*_10.tif"
    filename_regex = r".*_(?P<date>\d*)_.*"
    date_format = "%Y"

    # Fix this function because for SOME REASON it converts to int32...
    def _merge_files(self, filepaths, query):
        """Load and merge one or more files.
        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = int(round((query.maxx - query.minx) / self.res))
            out_height = int(round((query.maxy - query.miny) / self.res))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(out_shape=out_shape, window=from_bounds(*bounds, src.transform))
        else:
            dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res)

        tensor = torch.tensor(dest)  # type: ignore[attr-defined]
        return tensor


class CDL(tgCDL):
    filename_glob = "*_30m_cdls.tif"

    def __init__(
        self,
        root: str,
        years: Sequence[int] = [],
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = True,
        checksum: bool = False,
    ):
        if years:
            self.md5s = [m for m in self.md5s if m[0] in years]
        super().__init__(root, crs, res, transforms, cache, download, checksum)


class CDLMask(CDL):
    """
    Binary mask dataset based on the choice of a CDL index subset to serve as a positive indices.
    """

    def __init__(
        self,
        root: str,
        positive_indices: Sequence[int],
        years: Sequence[int] = [],
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = True,
        checksum: bool = False,
    ):
        super().__init__(root, years, crs, res, transforms, cache, download, checksum)
        self.positive_indices = torch.as_tensor(positive_indices)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        sample = super().__getitem__(query)
        sample["mask"] = torch.isin(sample["mask"], self.positive_indices).to(torch.float32)
        return sample
