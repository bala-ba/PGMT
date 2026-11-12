"""Render native Stage1 plane trajectories, with recorded reference poses."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plane_model():
    asset = ROOT / 'TCRS/assets/g1/g1_29dof_rev_1_0.xml'
    root = ET.parse(asset).getroot()
    root.find('compiler').set('meshdir', str(asset.parent / 'meshes'))
    for node in root.findall('visual'):
        root.remove(node)
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    ET.SubElement(visual, 'quality', shadowsize='2048', offsamples='4')
    ET.SubElement(visual, 'headlight', ambient='.25 .25 .25', diffuse='.35 .35 .35', specular='0 0 0')
    for world in root.findall('worldbody'):
        for node in list(world):
            if node.tag == 'light' or (node.tag == 'geom' and node.get('type') == 'plane'):
                world.remove(node)
    for assets in root.findall('asset'):
        for node in list(assets):
            if node.tag == 'texture' and node.get('type') == 'skybox':
                assets.remove(node)
    assets, world = root.find('asset'), root.find('worldbody')
    ET.SubElement(assets, 'texture', name='replay_sky', type='skybox', builtin='gradient',
                  rgb1='.64 .73 .81', rgb2='.88 .91 .94', width='512', height='3072')
    ET.SubElement(assets, 'texture', name='replay_floor', type='2d', builtin='checker',
                  rgb1='.34 .40 .46', rgb2='.43 .50 .56', width='512', height='512')
    ET.SubElement(assets, 'material', name='replay_floor', texture='replay_floor',
                  texrepeat='2 2', texuniform='true', reflectance='0', specular='0')
    ET.SubElement(world, 'geom', type='plane', size='0 0 .02', group='2', material='replay_floor', contype='0', conaffinity='0')
    ET.SubElement(world, 'light', pos='-2 -4 8', directional='true', dir='1 1 -3', diffuse='.55 .55 .55', specular='0 0 0')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / 'training_status.json').read_text())
    assert meta['status'] == 'completed' and meta['policy_family'] == 'pgmt_stage1'
    assert meta['policy_state_unchanged'] and meta['source_checkpoint_unchanged']
    assert sha(meta['checkpoint']) == meta['source_sha256']
    assert sha(meta['motion']) == meta['video_states']['source_motion_sha256']
    assert meta['terrain_perception'] is False and meta['random_pushes'] is False
    with np.load(source / 'video_states.npz') as archive:
        d = dict(archive)
    frames, times = d['frame'].astype(int), d['time_s']
    assert frames[0] == 0 and times[0] == 0 and d['terminal'][-1] and not d['terminal'][:-1].any()
    assert len(frames) == meta['steps'] + 1
    np.testing.assert_array_equal(frames[:-1], np.arange(len(frames)-1))
    np.testing.assert_allclose(np.diff(times), meta['control_dt'], atol=1e-7, rtol=0)
    assert all(np.isfinite(v).all() for v in d.values())
    names = meta['video_states']
    with np.load(meta['motion']) as motion:
        assert names['body_names'] == motion['body_names'].tolist()
        assert names['joint_names'] == motion['joint_names'].tolist()
        np.testing.assert_array_equal(d['reference_joint_pos'], motion['joint_pos'][frames])
        np.testing.assert_allclose(d['reference_body_pos_w'], motion['body_pos_w'][frames] + names['env_origin'], atol=1e-5, rtol=0)
        np.testing.assert_allclose(d['reference_body_quat_w'], motion['body_quat_w'][frames], atol=1e-6, rtol=0)
    model = plane_model()
    actual, reference = mujoco.MjData(model), mujoco.MjData(model)
    body_ids = [model.body(name).id for name in names['body_names']]
    joints = model.jnt_qposadr[[model.joint(name).id for name in names['joint_names']]]
    pelvis = names['body_names'].index('pelvis')
    origin = d['body_pos_w'][0, pelvis].copy()
    origin[2] = 0.
    offset = np.array([0., 1., 0.])
    options, ref_options = mujoco.MjvOption(), mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1:3] = 1
    ref_options.geomgroup[:] = 0
    ref_options.geomgroup[1] = 1
    perturb = mujoco.MjvPerturb()
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135., -25., 3.8
    renderer = mujoco.Renderer(model, width=960, height=640)
    fonts = {n: ImageFont.truetype(FONT, n) for n in (20, 22, 26)}
    motion_name = Path(meta['motion']).stem
    video = output / f'{motion_name}_stage1_flat.mp4'
    if video.exists():
        raise FileExistsError(video)
    fps, count = 25, int(np.ceil(times[-1]*25)) + 1
    errors = dict(actual=0., reference=0.)
    root_error = np.linalg.norm(d['body_pos_w'][:, pelvis, :2]-d['reference_body_pos_w'][:, pelvis, :2], axis=-1)
    def pose(state, index, prefix, shift, key):
        state.qpos[:3] = d[prefix+'body_pos_w'][index,pelvis]-origin+shift
        state.qpos[3:7] = d[prefix+'body_quat_w'][index,pelvis]
        state.qpos[joints] = d[prefix+'joint_pos'][index]
        state.qvel[:] = 0
        mujoco.mj_forward(model, state)
        expected = d[prefix+'body_pos_w'][index]-origin+shift
        error = float(np.linalg.norm(state.xpos[body_ids]-expected, axis=-1).max())
        errors[key] = max(errors[key], error)
        assert error < .001, (key, index, error)
    writer = imageio.get_writer(str(video), fps=fps, codec='libx264', quality=None, macro_block_size=16,
                               ffmpeg_params=['-crf','19','-preset','fast','-threads','2','-movflags','+faststart'])
    try:
        for fi in range(count):
            i = min(int(np.searchsorted(times, fi/fps)), len(times)-1)
            pose(actual, i, '', np.zeros(3), 'actual')
            pose(reference, i, 'reference_', offset, 'reference')
            camera.lookat[:] = (actual.qpos[:3]+reference.qpos[:3])/2 + [0., 0., -.06]
            camera.distance = max(3.8, 3.0+.65*np.linalg.norm(actual.qpos[:3]-reference.qpos[:3]))
            renderer.update_scene(actual, camera=camera, scene_option=options)
            scene = renderer.scene
            start = scene.ngeom
            mujoco.mjv_addGeoms(model, reference, ref_options, perturb, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scene)
            assert scene.ngeom > start
            for geom in scene.geoms[start:scene.ngeom]:
                geom.rgba[:], geom.matid, geom.emission = [.02,.72,.98,.72], -1, .15
            canvas = Image.new('RGB', (960,800), (13,20,30))
            canvas.paste(Image.fromarray(renderer.render()), (0,80))
            draw = ImageDraw.Draw(canvas)
            draw.text((18,6),f'Stage 1 · 平地跟踪 · {motion_name}',font=fonts[26],fill=(236,245,252))
            draw.text((18,44),f'{Path(meta["checkpoint"]).name} | t={times[i]:.2f}s | 参考帧 {frames[i]} | 1×',font=fonts[22],fill=(114,219,232))
            draw.rectangle((12,92,640,129), fill=(18,29,42))
            draw.text((22,95),'灰色：实际机器人 | 青蓝色：参考动作（横移 1 m）',font=fonts[20],fill=(133,228,247))
            draw.text((18,731),f'Stage 1 无视觉输入 | 无额外外力扰动 | 根部水平误差 {root_error[i]*100:.1f} cm',font=fonts[20],fill=(194,215,233))
            text = 'Isaac / PhysX 实际仿真轨迹；参考动作保持原始高度。'
            if i == len(times)-1:
                ending = meta['last_ending']
                text = ('提前终止' if ending['failed'] else '动作完成') + f' {times[-1]:.2f}s'
                if ending['failed']:
                    text += ' · '+','.join(k for k,v in ending['termination_flags'].items() if v)
            draw.text((18,766),text,font=fonts[20],fill=(153,220,183))
            writer.append_data(np.asarray(canvas))
            if fi in (0, count//2, count-1):
                canvas.save(output / f'preview_{fi:03d}.jpg', quality=94)
            if fi % 50 == 0:
                print(f'[STAGE1 VIDEO] {fi+1}/{count}', flush=True)
    finally:
        writer.close()
        renderer.close()
    encoded, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert encoded == count and abs(seconds-count/fps)<.03
    report = dict(video=str(video), video_sha256=sha(video), frames=count, fps=fps, video_seconds=seconds,
        simulation_seconds=float(times[-1]), resolution=[960,800], first_reference_frame=int(frames[0]),
        last_reference_frame=int(frames[-1]), checkpoint=meta['checkpoint'], checkpoint_sha256=meta['source_sha256'],
        motion=meta['motion'], motion_sha256=names['source_motion_sha256'], result=meta['last_ending'],
        terrain='native plane', terrain_perception=False, reference_enabled=True, reference_display_offset_m=offset.tolist(),
        reference_source_validation=True, max_rendered_body_error_m=errors,
        initial_joint_difference_max=float(abs(d['joint_pos'][0]-d['reference_joint_pos'][0]).max()),
        initial_root_xy_error_m=float(root_error[0]), root_xy_error_1s_m=float(root_error[np.argmin(abs(times-1.))]),
        root_xy_error_rmse_m=float(np.sqrt(np.mean(root_error**2))), root_xy_error_max_m=float(root_error.max()),
        physics='Isaac / PhysX', rendering='MuJoCo pose replay only; no mj_step',
        random_pushes=False, source_checkpoint_unchanged=True, policy_state_unchanged=True)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
