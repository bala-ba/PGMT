"""Adapt repeated native patches to birth-region curriculum and placement."""
import torch


PATCH_SCHEMA = "pmt.pgmt.terrain.patches.v1"
# Preserve the existing CSV's semantic groups and explicit inverted variants.
PATCH_COMPATIBILITY = {"flat": "flat", "stairs": "up_stairs", "slope": "up_slope",
                       "boxes": "boxes", "rough": "random_rough", "stairs_up": "stairs",
                       "stairs_down": "up_stairs", "platforms": "boxes"}
GROUP_IDS = {"flat": 0, "slope": 1, "stairs": 2, "boxes": 3, "rough": 4}
PATCH_GROUPS = {**{family: family for family in GROUP_IDS},
                "stairs_up": "stairs", "stairs_down": "stairs", "platforms": "boxes"}


def training_manifest(source):
    """Return an in-memory region view; never rewrite the audited asset."""
    if source.get("schema") != PATCH_SCHEMA:
        return source
    spec = source["spec"]
    if not spec["families"] or set(spec["families"]) - set(PATCH_COMPATIBILITY):
        raise ValueError("Unknown PGMT patch families")
    return dict(source, spec=dict(spec, size=spec["region_size"]), tiles=[
        dict(region, origin=list(region["center"]), family_group_index=GROUP_IDS[PATCH_GROUPS[region["family"]]])
        for region in source["regions"]])


def compatibility_families(manifest):
    families = manifest["spec"]["families"]
    if manifest.get("schema") == PATCH_SCHEMA:
        return tuple(PATCH_COMPATIBILITY[family] for family in families)
    return tuple(families)


def patch_bounds(manifest, *, device):
    """Validate complete regular patch coverage before sampling any starts."""
    spec = manifest["spec"]
    rows, cols = spec["num_rows"], spec["num_cols"]
    nr, nc = spec["patch_grid"]
    records = {(p["level"], p["family_index"], p["patch_id"]): p for p in manifest["patches"]}
    expected = {(r, c, p) for r in range(rows) for c in range(cols) for p in range(nr * nc)}
    if len(records) != len(manifest["patches"]) or records.keys() != expected:
        raise ValueError("PGMT patch manifest must cover every region and patch exactly once")
    regions = {(r["level"], r["family_index"]): r for r in manifest["regions"]}
    bounds = torch.empty(rows, cols, nr * nc, 2, 2, device=device)
    size = bounds.new_tensor(spec["patch_size"])
    if not torch.allclose(size * size.new_tensor([nr, nc]), size.new_tensor(spec["region_size"])):
        raise ValueError("Patch sizes must fill the region")
    for (r, c, p), record in records.items():
        actual = bounds.new_tensor(record["world_bounds"])[:, :2]
        lower = bounds.new_tensor(regions[r, c]["bounds"][0]) + size * size.new_tensor([p // nc, p % nc])
        if (record["patch_row"], record["patch_col"]) != (p // nc, p % nc) or not torch.allclose(
                actual, torch.stack((lower, lower + size)), atol=1e-4, rtol=0):
            raise ValueError(f"Malformed PGMT patch bounds: {record['name']}")
        bounds[r, c, p] = actual
    return bounds


def centered_patch_ids(patch_grid, spawn_grid, *, device):
    """Select complete central patches; reject ambiguous off-center grids."""
    if len(patch_grid) != 2 or len(spawn_grid) != 2:
        raise ValueError("Patch and spawn grids must each have two dimensions")
    for total, count in zip(patch_grid, spawn_grid):
        if (isinstance(total, bool) or isinstance(count, bool)
                or not isinstance(total, int) or not isinstance(count, int)
                or count <= 0 or total < count or (total - count) % 2):
            raise ValueError("Spawn grid must fit and be centered on complete patches")
    nr, nc = patch_grid
    sr, sc = spawn_grid
    row, col = (nr - sr) // 2, (nc - sc) // 2
    return torch.tensor([r * nc + c for r in range(row, row + sr)
                         for c in range(col, col + sc)], dtype=torch.long, device=device)


def sample_patch_positions(bounds, levels, families, *, margin, allowed_patch_ids=None, xy_mode="random"):
    """Select a patch, then its center or uniform XY; caller checks full-body poses."""
    if xy_mode not in ("center", "random"):
        raise ValueError(f"Unknown placement XY mode: {xy_mode!r}")
    if margin < 0 or (bounds[..., 1, :] - bounds[..., 0, :] <= 2 * margin).any():
        raise ValueError("Placement margin leaves no patch interior")
    if allowed_patch_ids is None:
        patch_ids = torch.randint(bounds.shape[2], levels.shape, device=levels.device)
    else:
        patch_ids = allowed_patch_ids[torch.randint(len(allowed_patch_ids), levels.shape, device=levels.device)]
    selected = bounds[levels, families, patch_ids]
    if xy_mode == "center":
        return selected.mean(dim=1), selected, patch_ids
    lower, upper = selected[:, 0] + margin, selected[:, 1] - margin
    xy = lower + (upper - lower) * torch.rand(len(levels), 2, device=bounds.device)
    return xy, selected, patch_ids
