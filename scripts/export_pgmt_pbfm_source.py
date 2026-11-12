"""Export P-BFM's original training_mix to mesh records for the Isaac builder.

Run with MJLab 1.4 / MuJoCo 3.8. The P-BFM repository is read-only input.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmt_tasks.pgmt_mujoco_mesh import collision_geom_mesh
from pmt_tasks.pgmt_terrain import ray_heights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed terrain asset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec_cfg = yaml.safe_load(Path(args.config).read_text())
    source = (ROOT / spec_cfg['source_repository']).resolve()
    sys.path.insert(0, str(source))
    versions = {name: importlib.metadata.version(name) for name in spec_cfg['source_versions']}
    if versions != spec_cfg['source_versions']:
        raise ValueError(f"P-BFM export requires {spec_cfg['source_versions']}, found {versions}")
    import mujoco
    from mjlab.terrains import TerrainGenerator
    from pbfm.terrains import presets
    from pbfm.terrains import primitives
    cfg = presets.make_training_mix_generator_cfg()
    names = list(cfg.sub_terrains)
    if (cfg.seed != spec_cfg['seed'] or names != spec_cfg['source_families']
            or list(cfg.size) != spec_cfg['patch_size'] or cfg.num_rows != spec_cfg['num_rows']
            or cfg.num_cols != spec_cfg['num_cols'] or cfg.curriculum != spec_cfg['curriculum']
            or list(cfg.difficulty_range) != spec_cfg['difficulty_range']
            or cfg.border_width != spec_cfg['border_width']):
        raise ValueError('P-BFM source preset differs from the declared export contract')
    for name, family in zip(names, spec_cfg['families']):
        if cfg.sub_terrains[name].proportion != spec_cfg['family_proportions'][family]:
            raise ValueError('P-BFM source training proportion differs')
    cells = []
    class RecordedGenerator(TerrainGenerator):
        def _create_terrain_geom(self, spec, world_position, difficulty, cfg, sub_row, sub_col):
            before = len(spec.body('terrain').geoms)
            origin = super()._create_terrain_geom(spec, world_position, difficulty, cfg, sub_row, sub_col)
            cells.append(dict(level=sub_row, family_index=sub_col, difficulty=float(difficulty),
                source_family=names[sub_col], family=spec_cfg['families'][sub_col], origin=origin.tolist(),
                geom_names=[f'terrain_{i}' for i in range(before, len(spec.body('terrain').geoms))]))
            return origin
    spec = mujoco.MjSpec()
    generator = RecordedGenerator(cfg)
    generator.compile(spec)
    guard_count, _ = presets.add_training_nonflat_guard_ring(spec, cfg)
    presets.add_training_outer_walls(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_saveModel(model, str(output/'source.mjb'), None)
    all_meshes = [collision_geom_mesh(model, data, i) for i in range(model.ngeom)]
    used = set()
    records, auxiliary = [], []
    for cell in sorted(cells, key=lambda c: (c['level'], c['family_index'])):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in cell.pop('geom_names')]
        if any(i < 0 for i in ids):
            raise ValueError('Source terrain geometry naming changed')
        used.update(ids)
        name = f"L{cell['level']:02d}_{cell['family']}"
        half_size = np.array(cfg.size)/2
        center = np.r_[(np.array([cell['level'], cell['family_index']]) + .5
                        - np.array([cfg.num_rows, cfg.num_cols])/2)*cfg.size, 0.]
        mesh = trimesh.util.concatenate([all_meshes[i] for i in ids])
        filename = name+'_P00.ply'
        mesh.apply_translation(-center)
        mesh.export(output/filename)
        bounds = mesh.bounds + center
        np.testing.assert_allclose(bounds[:, :2], [center[:2]-half_size, center[:2]+half_size], atol=1e-5)
        sub = asdict(cfg.sub_terrains[cell['source_family']])
        cell.update(name=name+'_P00',region_name=name,patch_id=0,patch_row=0,patch_col=0,
                    center=center.tolist(),world_bounds=bounds.tolist(),mesh_file=filename,
                    source_geom_ids=ids,parameters=dict(source_config=sub),
                    prim_path=f'/World/Terrain/{name}/P00',inverted=cell['family'] in ('stairs_down','slope'))
        records.append(cell)
    for i in range(model.ngeom):
        if i in used:
            continue
        name = model.geom(i).name
        group = 'Guard' if name.startswith('g1depth_guard_tile_') else ('Walls' if name.startswith('rp1_outer_wall_') else 'Border')
        mesh = all_meshes[i]
        center = mesh.bounds.mean(0)
        filename = f'aux_{i:04d}.ply'
        centered = mesh.copy(); centered.apply_translation(-center); centered.export(output/filename)
        auxiliary.append(dict(name=name,family=group.lower(),center=center.tolist(),world_bounds=mesh.bounds.tolist(),
                              mesh_file=filename,source_geom_ids=[i],prim_path=f'/World/Terrain/{group}/G{i:04d}'))
    # Reference rays use the FULL native model: the original underlay can cover
    # negative parts of guard heightfields, which must also hold in Isaac.
    combined = trimesh.util.concatenate(all_meshes)
    rng = np.random.default_rng(20261111)
    max_error, ray_count = 0., 0
    top = float(combined.bounds[1,2]+5)
    for record in records+auxiliary:
        bounds = np.array(record['world_bounds'])[:, :2]
        n = 512 if 'level' in record else 128
        xy = rng.uniform(bounds[0]+.0007, bounds[1]-.0007, size=(n,2))
        expected = []
        for x,y in xy:
            dist = mujoco.mj_ray(model,data,np.array([x,y,top]),np.array([0.,0.,-1.]),None,1,-1,None)
            if dist < 0:
                raise ValueError(f"Native MuJoCo ray missed {record['name']}")
            expected.append(top-dist)
        actual = ray_heights(combined,xy)
        error = float(np.abs(actual-expected).max())
        if error > 2e-5:
            worst = int(np.abs(actual-expected).argmax())
            raise ValueError(f"MuJoCo/mesh mismatch {record['name']}: {error} at {xy[worst]}, mesh={actual[worst]}, MuJoCo={expected[worst]}")
        max_error=max(max_error,error); ray_count+=n
        record['world_probes']=np.column_stack((xy,expected)).tolist()
        record['mesh_sha256']=hashlib.sha256((output/record['mesh_file']).read_bytes()).hexdigest()
        if 'level' in record:
            preview = spec_cfg['spawn_preview']
            center=np.array(record['center'])
            xy=rng.uniform(center[:2]-half_size+preview['boundary_margin'],center[:2]+half_size-preview['boundary_margin'],
                           size=(preview['samples_per_patch'],2))
            record['preview_xyzyaw']=np.column_stack((xy,ray_heights(combined,xy),rng.uniform(*preview['yaw_range'],len(xy)))).tolist()
        print(f"PASS {record['name']}: MuJoCo/mesh {error:.3g} m",flush=True)
    sources={Path(presets.__file__),Path(primitives.__file__),Path(sys.modules[TerrainGenerator.__module__].__file__)}
    sources.update(Path(module.__file__) for name,module in list(sys.modules.items())
                   if name.startswith('mjlab.terrains') and getattr(module,'__file__',None))
    half_extent = np.array(cfg.size)*[cfg.num_rows,cfg.num_cols]/2
    metadata=dict(spec=spec_cfg,patches=records,auxiliary_surfaces=auxiliary,guard_count=guard_count,
        source=dict(repository=str(source),commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                    versions=versions,preset=asdict(cfg),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
                    model_sha256=hashlib.sha256((output/'source.mjb').read_bytes()).hexdigest()),
        validation=dict(mujoco_mesh_rays=ray_count,max_mujoco_mesh_error_m=max_error),
        walkable_bounds=[(-half_extent-cfg.border_width).tolist(),(half_extent+cfg.border_width).tolist()],
        map_bounds=[(-half_extent).tolist(),half_extent.tolist()])
    (output/'source_export.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f"SOURCE EXPORT PASSED: {len(records)} training + {len(auxiliary)} auxiliary meshes",flush=True)


if __name__ == '__main__':
    main()
