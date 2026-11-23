"""Draw point clouds from predicted PPP fields (CPU, float64, dedicated generator).

TODO: DO PROPER POISSON SAMPLING

Events sit at cell centers, several per cell are kept. The grid convention matches
c3pdar.data.raster_cells: row 0 is the largest y.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .data import raster_cells


@dataclass
class Raster:
    """A predicted marked intensity over [-point_limit, point_limit]^2 as (H,W) float64 CPU tensors: expected
    point count, RCS (dBsm) and Doppler (m/s) per cell.

    Cells follow raster_cells. ``shift`` moves all cell centers by that many cells towards -x and -y:
    RadarGen's maps center pixel column j at x = -L + j * cell size (shift 0.5), ours at -L + (j + 0.5) * size.
    """

    masses: torch.Tensor
    rcs: torch.Tensor
    doppler: torch.Tensor
    point_limit: float
    shift: float = 0.0

    @property
    def cell_size(self):
        """[dx, dy] in meters."""
        h, w = self.masses.shape
        return np.array([2 * self.point_limit / w, 2 * self.point_limit / h])

    def cells(self, xy):
        """Row and column (numpy) of the cell containing each (N,2) [x,y]; points outside are clamped."""
        xy = torch.as_tensor(np.asarray(xy, dtype=np.float64).reshape(-1, 2) + self.shift * self.cell_size)
        limit = self.point_limit
        rows, cols, _ = raster_cells(xy, self.masses.shape, -limit, limit, -limit, limit)
        return rows.numpy(), cols.numpy()

    def centers(self, rows, cols):
        """(len(rows)*len(cols),2) [x,y] of the cells rows x cols, row-major."""
        dx, dy = self.cell_size
        y = (self.masses.shape[0] - np.asarray(rows) - 0.5 - self.shift) * dy - self.point_limit
        x = (np.asarray(cols) + 0.5 - self.shift) * dx - self.point_limit
        y, x = np.meshgrid(y, x, indexing="ij")
        return np.column_stack([x.ravel(), y.ravel()])


def ppp_raster(fields, point_limit, mark_link="cubic"):
    """Raster of our raw (3,H,W) fields [log-intensity, RCS, Doppler]."""
    fields = torch.as_tensor(fields)
    return Raster(cell_masses(fields[0]), *physical_marks(fields[1:], mark_link), point_limit)


def density_raster(maps, count, point_limit):
    """Raster of RadarGen's (3,H,W) maps [point density, RCS, Doppler] with ``count`` expected points.

    The density map sums to 1 and its scale is lost in the max-normalized image, so the count comes from
    the deconvolved cloud.
    """
    maps = torch.as_tensor(maps, dtype=torch.float64)
    return Raster(count * maps[0] / maps[0].sum(), maps[1], maps[2], point_limit, shift=0.5)


def cell_masses(log_intensity):
    """(H,W) expected number of points per cell, exp(log-intensity) / (H*W), float64 on CPU."""
    grid = log_intensity.detach().to(device="cpu", dtype=torch.float64)
    return grid.exp() / grid.numel()


def physical_marks(raw, mark_link="cubic"):
    raw = raw.detach().to(device="cpu", dtype=torch.float64)
    return raw.pow(3) if mark_link == "cubic" else raw


def cell_centers(shape, point_limit):
    """(H*W,2) [x,y] meters of all cell centers in row-major order."""
    h, w = shape
    rows, cols = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64),
                                indexing="ij")
    x = (cols + 0.5) / w * (2 * point_limit) - point_limit
    y = (h - rows - 0.5) / h * (2 * point_limit) - point_limit
    return torch.stack((x.flatten(), y.flatten()), dim=1)


@torch.no_grad()
def sample_points(masses, rcs, doppler, point_limit, generator, scale=1.0):
    """Float64 (N,4) [x, y, RCS, Doppler] with Poisson(scale * mass) events per cell.

    ``masses`` from cell_masses, marks already physical. ``scale`` rescales the intensity,
    e.g. budget / mean expected count over the evaluated set.
    """
    if generator.device.type != "cpu":
        raise ValueError("Supply a seeded CPU torch.Generator")
    counts = torch.poisson(masses * scale if scale != 1.0 else masses, generator=generator)
    indices = torch.repeat_interleave(torch.arange(masses.numel()), counts.flatten().long())
    if indices.numel() == 0:
        return np.empty((0, 4), dtype=np.float64)
    xy = cell_centers(masses.shape, float(point_limit))[indices]
    return torch.cat((xy, rcs.flatten()[indices, None], doppler.flatten()[indices, None]), dim=1).numpy()


def sample_generator(seed, index):
    """Per-frame CPU generator, independent of evaluation order (as in RadarGen's validation)."""
    return torch.Generator().manual_seed(int(np.random.SeedSequence([seed, index]).generate_state(1)[0]))


def draw_generator(seed, scene_token, frame_index, draw):
    """CPU generator of one draw of one frame, independent of batching, frame selection and order."""
    scene = int.from_bytes(hashlib.sha256(scene_token.encode()).digest()[:8], "little")
    state = np.random.SeedSequence([seed, scene, int(frame_index), draw]).generate_state(1, np.uint64)[0]
    return torch.Generator().manual_seed(int(state))
