"""Convert static MuJoCo collision boxes/heightfields without resampling them.

MuJoCo is imported lazily so Isaac training only needs the exported USD/PLY.
"""
import numpy as np
import trimesh


def heightfield_mesh(heights, size):
    """MuJoCo's row-major samples, triangle diagonal, side walls and base."""
    rows, cols = heights.shape
    sx, sy, sz, base = size
    xx, yy = np.meshgrid(np.linspace(-sx, sx, cols), np.linspace(-sy, sy, rows))
    vertices = np.column_stack((xx.ravel(), yy.ravel(), (heights * sz).ravel()))
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b, d, e = a + 1, a + cols, a + cols + 1
    faces = np.vstack((np.column_stack((a, b, e)), np.column_stack((a, e, d))))
    # Counterclockwise perimeter, viewed from above.
    rim = np.r_[np.arange(cols), np.arange(1, rows) * cols + cols - 1,
                (rows - 1) * cols + np.arange(cols - 2, -1, -1), np.arange(rows - 2, 0, -1) * cols]
    bottom = len(vertices) + np.arange(len(rim))
    vertices = np.vstack((vertices, np.column_stack((vertices[rim, :2], np.full(len(rim), -base)))))
    nxt = np.roll(np.arange(len(rim)), -1)
    faces = np.vstack((faces, np.column_stack((rim, bottom, bottom[nxt])),
                       np.column_stack((rim, bottom[nxt], rim[nxt])),
                       np.column_stack((np.full(len(rim) - 2, bottom[0]), bottom[2:], bottom[1:-1]))))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def collision_geom_mesh(model, data, geom_id):
    import mujoco
    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * model.geom_size[geom_id])
    elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
        field = model.geom_dataid[geom_id]
        rows, cols = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
        start = model.hfield_adr[field]
        heights = model.hfield_data[start:start + rows * cols].reshape(rows, cols)
        mesh = heightfield_mesh(heights, model.hfield_size[field])
    else:
        raise ValueError(f"Unsupported P-BFM collision geometry: {model.geom(geom_id).name}, type={kind}")
    transform = np.eye(4)
    transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
    transform[:3, 3] = data.geom_xpos[geom_id]
    mesh.apply_transform(transform)
    return mesh
