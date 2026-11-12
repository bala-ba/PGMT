"""Position-based tile lookup, independent of environment curriculum assignment."""
import torch


class TerrainTileMap:
    """Index the manifest's regular XY grid without scanning all tiles per point.

    Rows run along X, columns along Y. Shared edges belong to the tile on the
    positive side; the outer upper edges are excluded. The surrounding border
    is outside the valid training map. Lookups never modify spawn assignments.
    """

    def __init__(self, manifest, tile_bounds):
        spec = manifest["spec"]
        self.rows, self.cols = spec["num_rows"], spec["num_cols"]
        records = {(t["level"], t["family_index"]): t for t in manifest["tiles"]}
        expected_keys = {(r, c) for r in range(self.rows) for c in range(self.cols)}
        if len(records) != len(manifest["tiles"]) or records.keys() != expected_keys:
            raise ValueError("PGMT tile manifest must contain each row/column exactly once")
        if tile_bounds.shape != (self.rows, self.cols, 2, 2):
            raise ValueError("PGMT tile bounds do not match the manifest grid")
        self.bounds = tile_bounds
        self.size = tile_bounds.new_tensor(spec["size"])
        centers = tile_bounds.new_tensor([[records[r, c]["center"][:2]
                                          for c in range(self.cols)] for r in range(self.rows)])
        self.lower = centers[0, 0] - self.size / 2
        self.upper = self.lower + self.size * self.size.new_tensor([self.rows, self.cols])
        self.x_edges = self.lower[0] + torch.arange(self.rows + 1, device=tile_bounds.device) * self.size[0]
        self.y_edges = self.lower[1] + torch.arange(self.cols + 1, device=tile_bounds.device) * self.size[1]
        row, col = torch.meshgrid(torch.arange(self.rows, device=tile_bounds.device),
                                 torch.arange(self.cols, device=tile_bounds.device), indexing="ij")
        cell_lower = self.lower + torch.stack((row, col), -1) * self.size
        if not torch.all(self.size > 0) or not torch.allclose(centers, cell_lower + self.size / 2, atol=1e-5, rtol=0):
            raise ValueError("PGMT tile centers must form the manifest's regular XY grid")
        if (not torch.isfinite(tile_bounds).all() or (tile_bounds[..., 0, :] >= tile_bounds[..., 1, :]).any()
                or (tile_bounds[..., 0, :] < cell_lower - 1e-4).any()
                or (tile_bounds[..., 1, :] > cell_lower + self.size + 1e-4).any()):
            raise ValueError("PGMT mesh bounds must be finite and contained in their grid cells")
        self.group_ids = torch.tensor([[records[r, c]["family_group_index"] for c in range(self.cols)]
                                       for r in range(self.rows)], dtype=torch.long, device=tile_bounds.device)
        self.difficulties = tile_bounds.new_tensor([[records[r, c]["difficulty"] for c in range(self.cols)]
                                                    for r in range(self.rows)])
        self.patch_grid = spec.get("patch_grid")
        if self.patch_grid:
            nr, nc = self.patch_grid
            self.patch_x_edges = self.lower[0] + torch.arange(self.rows * nr + 1, device=tile_bounds.device) * (self.size[0] / nr)
            self.patch_y_edges = self.lower[1] + torch.arange(self.cols * nc + 1, device=tile_bounds.device) * (self.size[1] / nc)

    def _indices(self, xy):
        if xy.shape[-1] != 2 or xy.device != self.bounds.device:
            raise ValueError("Terrain queries require world XY on the terrain device")
        finite = torch.isfinite(xy).all(-1)
        # Search exact edges: arithmetic floor((xy-lower)/size) can round a
        # point just below a seam into the next tile, especially at negative XY.
        row = torch.bucketize(xy[..., 0].contiguous(), self.x_edges, right=True) - 1
        col = torch.bucketize(xy[..., 1].contiguous(), self.y_edges, right=True) - 1
        valid = finite & (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)
        row, col = row.clamp(0, self.rows - 1), col.clamp(0, self.cols - 1)
        bounds = self.bounds[row, col]
        valid &= (xy >= bounds[..., 0, :]).all(-1) & (xy <= bounds[..., 1, :]).all(-1)
        return row, col, valid

    def contains(self, xy):
        """Geometric map membership; no per-tile reset margin or border padding."""
        return self._indices(xy)[2]

    def locate(self, xy):
        """Return actual tile metadata for arbitrary [...,2] world coordinates.

        tile_level describes geometry under the query point, not curriculum.
        Invalid points get -1 indices and NaN difficulty, never a clamped tile.
        """
        row, col, valid = self._indices(xy)
        result = dict(tile_valid=valid, tile_level=torch.where(valid, row, -1),
                    family_index=torch.where(valid, col, -1),
                    family_group_index=torch.where(valid, self.group_ids[row, col], -1),
                    difficulty=torch.where(valid, self.difficulties[row, col], float("nan")))
        if self.patch_grid:
            nr, nc = self.patch_grid
            pr = (torch.bucketize(xy[..., 0].contiguous(), self.patch_x_edges, right=True) - 1) % nr
            pc = (torch.bucketize(xy[..., 1].contiguous(), self.patch_y_edges, right=True) - 1) % nc
            result["patch_id"] = torch.where(valid, pr * nc + pc, -1)
        return result
