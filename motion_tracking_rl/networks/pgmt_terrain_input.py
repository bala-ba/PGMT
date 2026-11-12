"""Network-side PGMT map preprocessing matching PMT's vision input path.

Terrain-Glimpse/IFM will consume this module in P2.2/P2.3. The Stage-1 actor
remains blind; this module does not add terrain tokens or change its checkpoint.
"""
from torch import nn


class PGMTTerrainInput(nn.Module):
    def __init__(self, z_clip=3.0, normalize_height=True):
        super().__init__()
        self.z_clip = float(z_clip)
        self.normalize_height = bool(normalize_height)

    def forward(self, terrain_map, terrain_valid):
        if terrain_map.ndim != 3 or terrain_map.shape != terrain_valid.shape:
            raise ValueError("PGMT network terrain map/mask must be matching [N,H,W] tensors")
        height = terrain_map
        if self.z_clip > 0.0:
            height = height.clamp(-self.z_clip, self.z_clip)
            if self.normalize_height:
                height = height / self.z_clip
        valid = (terrain_valid > 0.5).to(dtype=height.dtype)
        # PMT keeps invalid height sentinels alongside a mask input channel.
        return height, valid
