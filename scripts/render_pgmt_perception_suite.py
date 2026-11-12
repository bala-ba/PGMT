"""One video per motion, plus individual terrain clips, with recorded policy vision."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_pgmt_force_comparison as replay
from pgmt_force_analysis import load_run

LABELS = dict(flat='FLAT / 平地',rough='ROUGH / 粗糙地面',up_stair='UP STAIR / 上楼梯',
    down_stair='DOWN STAIR / 下楼梯',up_slope='UP SLOPE / 上坡',down_slope='DOWN SLOPE / 下坡',boxes='BOXES / 箱体')
MOTIONS = {'walk1_subject1_clip1':'步行 WALK','run2_subject1_clip0':'跑步 RUN','jumps1_subject1_clip0':'跳跃 JUMP'}
FONTS = {s:ImageFont.truetype(replay.FONT,s) for s in (14,16,18,20,22,26,32)}


def colors(values, valid):
    from matplotlib import colormaps
    rgba = colormaps['viridis'](np.clip((values+.5)/2.,0,1))
    rgb = (rgba[...,:3]*255).astype(np.uint8)
    rgb[~valid] = (240,75,180)
    return rgb


def panel(d, i):
    im=Image.new('RGB',(480,240),(19,29,41)); draw=ImageDraw.Draw(im)
    draw.text((12,3),'策略高度图 / m · 前方 ↑',font=FONTS[16],fill=(220,233,245))
    h,v=d['terrain_map'][i],d['terrain_valid'][i].astype(bool)
    # Actual torso +X is up; +Y (left) is left. Stored array is row Y/col X.
    heat=colors(h,v).transpose(1,0,2)[::-1,::-1]
    im.paste(Image.fromarray(heat).resize((170,170),Image.Resampling.NEAREST),(15,29))
    for k,(x,y) in enumerate(d['center_xy_m'][i]):
        px,py=15+(1-y)*85,29+(1-x)*85
        draw.rectangle((px-17,py-17,px+17,py+17),outline=(255,100,223),width=2)
        draw.text((px-4,py-10),str(k+1),font=FONTS[14],fill=(255,196,70))
    draw.text((15,202),'-0.5  →  1.5 m',font=FONTS[14],fill=(170,188,207))
    draw.text((205,3),'4 个感知块 · 0.4 × 0.4 m',font=FONTS[16],fill=(225,232,244))
    for k in range(4):
        x,y=207+k*66,29
        rgb=colors(d['patch_height'][i,k]*3,d['patch_valid'][i,k].astype(bool)).transpose(1,0,2)[::-1,::-1]
        im.paste(Image.fromarray(rgb).resize((58,58),Image.Resampling.NEAREST),(x,y))
        draw.text((x,y+61),f"{k+1}: {d['coverage'][i,k]*100:.0f}%",font=FONTS[14],fill=(193,209,226))
    mass=d['terrain_attention_mass'][i].flatten()
    draw.text((205,118),'地形注意力 / 4 heads',font=FONTS[16],fill=(215,230,242))
    for k,value in enumerate(mass):
        x,y=208+k*65,145
        draw.rectangle((x,y,x+53,y+8),fill=(55,67,80))
        draw.rectangle((x,y,x+53*float(value),y+8),fill=(255,174,72))
        draw.text((x,y+12),f'{value:.3f}',font=FONTS[14],fill=(211,221,232))
    draw.text((205,190),f"有效扫描：{int(v.sum())}/441",font=FONTS[16],fill=(98,222,228))
    draw.text((205,213),f"观测时刻：{d['input_time_s'][i]:.2f} s",font=FONTS[14],fill=(171,191,209))
    return im


class VisionView(replay.View):
    def enable_reference(self):
        self.reference_state=mujoco.MjData(self.model)
        self.reference_options=mujoco.MjvOption()
        self.reference_options.geomgroup[:]=0
        self.reference_options.geomgroup[1]=1
        self.reference_perturb=mujoco.MjvPerturb()
        self.reference_offset=np.array([0.,1.,0.])
        self.reference_max_error=0.
        self.reference_quat_error=0.

    def render(self,index):
        d,state=self.d,self.state
        state.qpos[:3]=d['body_pos_w'][index,self.root_id]-self.origin
        state.qpos[3:7]=d['body_quat_w'][index,self.root_id]
        state.qpos[self.joints]=d['joint_pos'][index]
        state.qvel[:]=0
        mujoco.mj_forward(self.model,state)
        error=float(np.linalg.norm(state.xpos[self.body_ids]-(d['body_pos_w'][index]-self.origin),axis=-1).max())
        self.max_error=max(self.max_error,error)
        if error>.001:raise ValueError(f'Pose mismatch {error} m')
        self.camera.lookat[:]=state.qpos[:3]+np.array([.12,0,-.08])
        if hasattr(self,'reference_state'):
            ref=self.reference_state
            ref.qpos[:3]=d['reference_body_pos_w'][index,self.root_id]-self.origin+self.reference_offset
            ref.qpos[3:7]=d['reference_body_quat_w'][index,self.root_id]
            ref.qpos[self.joints]=d['reference_joint_pos'][index]
            ref.qvel[:]=0
            mujoco.mj_forward(self.model,ref)
            expected=d['reference_body_pos_w'][index]-self.origin+self.reference_offset
            error=float(np.linalg.norm(ref.xpos[self.body_ids]-expected,axis=-1).max())
            expectedq=d['reference_body_quat_w'][index]
            actualq=ref.xquat[self.body_ids]
            qerror=float(np.minimum(np.linalg.norm(actualq-expectedq,axis=-1),np.linalg.norm(actualq+expectedq,axis=-1)).max())
            self.reference_max_error=max(self.reference_max_error,error)
            self.reference_quat_error=max(self.reference_quat_error,qerror)
            if error>.001 or qerror>1e-4:raise ValueError(f'Reference FK mismatch {error} m / {qerror}')
            self.camera.lookat[:]=(state.qpos[:3]+ref.qpos[:3])/2+np.array([.12,0,-.08])
            self.camera.distance=max(3.8,3.0+.65*np.linalg.norm(ref.qpos[:3]-state.qpos[:3]))
            self.camera.azimuth=135.
        self.renderer.update_scene(state,camera=self.camera,scene_option=self.options)
        scene=self.renderer.scene
        if hasattr(self,'reference_state'):
            start=scene.ngeom
            mujoco.mjv_addGeoms(self.model,self.reference_state,self.reference_options,
                self.reference_perturb,int(mujoco.mjtCatBit.mjCAT_DYNAMIC),scene)
            if scene.ngeom==start:raise ValueError('No reference robot visual geometry')
            for geom in scene.geoms[start:scene.ngeom]:
                geom.rgba[:]=[.02,.72,.98,.72]
                geom.matid=-1
                geom.emission=.15
                geom.specular=.1
                geom.reflectance=0
        def sphere(pos,radius,color):
            if scene.ngeom>=scene.maxgeom:raise ValueError('Overlay capacity')
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3,radius),pos,np.eye(3).ravel(),np.array(color,dtype=float))
            scene.ngeom+=1
        valid=d['terrain_valid'][index].flatten().astype(bool)
        hits=d['ray_hits_w'][index]-self.origin
        for k in range(len(hits)):
            if np.isfinite(hits[k]).all():
                sphere(hits[k]+[0,0,.018],.009,(.05,.95,1,1) if valid[k] else (1,.1,.5,1))
        for center in d['centers_w'][index]-self.origin:
            sphere(center,.038,(1,.55,.02,1))
        for border in d['borders_w'][index]-self.origin:
            for a,b in zip(border[:-1],border[1:]):
                geom=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.05,.65,1]))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,3.0,a,b)
                scene.ngeom+=1
        return Image.fromarray(self.renderer.render())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wait_for_trials',action='store_true')
    ap.add_argument('--preview_only',action='store_true')
    ap.add_argument('--show_reference',action='store_true',help='Render cyan reference robot at world Y +1m for comparison')
    ap.add_argument('--individual_only',action='store_true',help='Export only separate terrain videos, with no overview montage')
    args=ap.parse_args(); args.source=args.source.resolve(); args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    suite=json.loads((args.source/'suite.json').read_text())
    checkpoint_name=Path(suite['checkpoint']).name
    if len(suite['levels'])!=1:raise ValueError('Render one terrain level at a time')
    level=suite['levels'][0]
    replay.PANEL,replay.RENDER_H=960,640
    results=[]; validations=[]
    for mi,motion in enumerate(suite['motions']):
        trials=[t for t in suite['trials'] if t['motion_index']==mi]
        if not 1<=len(trials)<=7:raise ValueError('Need one to seven ordered terrains per motion')
        if args.wait_for_trials:
            while True:
                ready=all((args.source/t['directory']/'perception.npz').is_file() and
                    json.loads((args.source/t['directory']/'training_status.json').read_text())['status']=='completed' for t in trials)
                if ready:break
                status=json.loads((args.source/'training_status.json').read_text())
                if status['status'] in ('failed','failed_physics','completed'):raise RuntimeError('Acquisition incomplete')
                time.sleep(5)
        out=args.output/motion['name']; out.mkdir(exist_ok=True)
        meshes=out/'render_assets'; meshes.mkdir(exist_ok=True)
        runs=[]; data=[]; views=[]; shared={}; errors={}; writers=[]; paths=[]; counts=[]
        reference_proofs={}
        overview=None
        try:
            for trial in trials:
                run=load_run(args.source/trial['directory'])
                with np.load(args.source/trial['directory']/'perception.npz') as archive:d=dict(archive)
                assert run['metadata']['source_sha256']==suite['source_sha256']
                assert len(d['time_s'])==run['metadata']['steps']+1
                assert d['terminal'][-1] and not d['terminal'][:-1].any()
                for key in ('terrain_map','center_xy_m','attention_weights','body_pos_w','joint_pos'):
                    assert np.isfinite(d[key]).all(),key
                # Exact terminal pose must come from the old episode, never its reset.
                np.testing.assert_array_equal(d['body_pos_w'][-1],run['data']['body_pos_w'][-1])
                if args.show_reference:
                    from pgmt_reference_replay import load_recorded_reference
                    reference,proof=load_recorded_reference(run,d)
                    d.update(reference)
                    reference_proofs[trial['label']]=proof
                if trial['terrain'] not in shared:
                    shared[trial['terrain']]=VisionView(run,meshes)
                    if args.show_reference:shared[trial['terrain']].enable_reference()
                view=shared[trial['terrain']]
                view.camera.distance=3.2; view.camera.azimuth=112; view.camera.elevation=-28
                views.append(view);runs.append(run);data.append(d)
                results.append({**trial, 'motion':motion['name'], **run['metadata']['last_episode']})
                paths.append(out/(Path(trial['directory']).name+'.mp4'))
                counts.append(int(np.ceil(float(d['time_s'][-1])*25))+1)
            overview_path=out/'00_seven_terrains_vision.mp4'
            video_paths=paths if args.individual_only else [overview_path,*paths]
            if any(p.exists() for p in video_paths) and not args.preview_only:
                raise FileExistsError('Refusing to overwrite videos')
            def writer(p):
                return imageio.get_writer(str(p),fps=25,codec='libx264',quality=None,macro_block_size=16,
                    ffmpeg_params=['-crf','19','-preset','fast','-threads','2','-movflags','+faststart'])
            if not args.preview_only:
                if not args.individual_only:overview=writer(overview_path)
                writers=[writer(p) for p in paths]
            frames=sorted({0,max(counts)//2,max(counts)-1}) if args.preview_only else range(max(counts))
            cached={}
            for fi in frames:
                stamp=fi/25
                canvas=Image.new('RGB',(3360,768),(13,20,30)); draw=ImageDraw.Draw(canvas)
                motion_label=motion.get('label',MOTIONS.get(motion['name'],'动作'))
                title=f"Level {level} · {motion_label} · {motion['name']}"
                draw.text((20,7),title,font=FONTS[32],fill=(241,247,252))
                subtitle=('灰色：实际机器人 | 青蓝色：参考动作（横移 1 m）| 视觉感知可视化 | 1× 播放'
                    if args.show_reference else f'{checkpoint_name} | 真实策略高度扫描 + 4 个局部感知块 | 1× 播放')
                draw.text((20,49),subtitle,font=FONTS[20],fill=(168,192,215))
                draw.text((3130,17),f't = {stamp:.2f} s',font=FONTS[26],fill=(225,238,248))
                for col,(trial,run,d,view) in enumerate(zip(trials,runs,data,views)):
                    end=float(d['time_s'][-1]); stopped=stamp>=end-1e-8
                    index=min(int(np.searchsorted(d['time_s'],stamp,side='left')),len(d['time_s'])-1)
                    key=(col,index)
                    if key not in cached:
                        view.d=d;view.max_error=0
                        picture=view.render(index); info=panel(d,index)
                        errors[trial['label']]=max(errors.get(trial['label'],0),view.max_error)
                        if index==len(d['time_s'])-1:cached[key]=(picture,info)
                    else:picture,info=cached[key]
                    x=480*col
                    draw.rectangle((x,80,x+478,143),fill=(25,37,51))
                    draw.text((x+12,85),LABELS[trial['label']],font=FONTS[22],fill=(120,221,233))
                    draw.text((x+12,116),'青：扫描点  橙：中心  粉：感知块',font=FONTS[16],fill=(190,208,224))
                    canvas.paste(picture.resize((480,320),Image.Resampling.LANCZOS),(x,144))
                    canvas.paste(info,(x,464))
                    result=run['metadata']['last_episode']
                    terminal_text=''
                    color=(132,233,179)
                    if stopped:
                        terminal_text=('提前终止' if result['failed'] else '动作完成')+f" {end:.2f}s"
                        if result['failed']:
                            color=(255,132,125)
                            terminal_text+=' · '+','.join(k for k,v in result['termination_flags'].items() if v and k not in ('time_out','motion_finished'))
                        draw.rectangle((x+5,422,x+474,460),fill=(25,29,38))
                        draw.text((x+13,427),terminal_text,font=FONTS[18],fill=color)
                    if not args.preview_only and fi<counts[col]:
                        single=Image.new('RGB',(960,960),(13,20,30)); sd=ImageDraw.Draw(single)
                        sd.text((18,5),title,font=FONTS[26],fill=(236,245,252))
                        sd.text((18,42),f"{LABELS[trial['label']]} | t={min(stamp,end):.2f}s | {checkpoint_name}",font=FONTS[22],fill=(114,219,232))
                        single.paste(picture,(0,80)); single.paste(info,(0,720))
                        if args.show_reference:
                            sd.rectangle((12,92,590,129),fill=(18,29,42))
                            sd.text((22,95),'灰色：实际  |  青蓝色：参考（横移 1 m）',font=FONTS[20],fill=(133,228,247))
                        sd.text((500,736),'真实策略输入 · 21×21 · 2×2 m',font=FONTS[20],fill=(211,228,244))
                        sd.text((500,779),'青色：实际高度扫描点',font=FONTS[20],fill=(90,226,237))
                        sd.text((500,814),'橙色：4 个感知块中心',font=FONTS[20],fill=(255,187,88))
                        sd.text((500,849),'粉色：0.4 m 感知块边界',font=FONTS[20],fill=(253,119,221))
                        if terminal_text:sd.text((500,895),terminal_text,font=FONTS[18],fill=color)
                        if args.individual_only and fi in (0,counts[col]//2,counts[col]-1):
                            single.save(out/f"{Path(trial['directory']).name}_preview_{fi:03d}.jpg",quality=94)
                        writers[col].append_data(np.asarray(single))
                    draw.line((x+479,80,x+479,704),fill=(73,87,104),width=2)
                footer=('参考使用同一帧和原始回合变换，仅横移展示；保持原始动作高度，可与地形相交。真实仿真轨迹和视觉感知保持原记录；结束后保持末帧。'
                    if args.show_reference else '实际 Isaac / PhysX 仿真；仅回放渲染。高度图与感知块来自该步策略输入。结束后保持末帧，终止帧的观测保持最后一个控制输入。')
                draw.text((20,717),footer,font=FONTS[18],fill=(168,190,211))
                if not args.individual_only and fi in (0,max(counts)//2,max(counts)-1):canvas.save(out/f'preview_{fi:03d}.jpg',quality=94)
                if overview is not None:overview.append_data(np.asarray(canvas))
                if fi%50==0:print(f"[VISION {motion['name']}] {fi+1}/{max(counts)}",flush=True)
            for w in writers:w.close()
            writers=[]
            if overview is not None:overview.close();overview=None
            if not args.preview_only:
                checked=[]
                expected_counts=counts if args.individual_only else [max(counts),*counts]
                for path,expected in zip(video_paths,expected_counts):
                    n,seconds=imageio_ffmpeg.count_frames_and_secs(str(path))
                    assert n==expected and abs(seconds-n/25)<.03
                    checked.append(dict(file=str(path),frames=n,seconds=seconds,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                validation=dict(motion=motion['name'],level=level,terrain_order=[t['label'] for t in trials],
                    videos=checked,fps=25,overview_resolution=None if args.individual_only else [3360,768],individual_resolution=[960,960],
                    max_fk_error_m=errors,perception_source='recorded exact pre-action observation and policy debug',
                    perception_enabled=True,terminal_input_hold_max_s=.02,physics='Isaac / PhysX',
                    renderer='MuJoCo pose replay only; no mj_step',checkpoint_sha256=suite['source_sha256'])
                if args.show_reference:
                    validation['reference_visualization']=dict(enabled=True,display_offset_m=[0,1,0],
                        terrain_height_adjustment=False,physics_integration=False,
                        source_validation=reference_proofs,
                        max_rendered_body_error_m=max(v.reference_max_error for v in shared.values()),
                        max_rendered_quaternion_error=max(v.reference_quat_error for v in shared.values()))
                validations.append(validation)
                (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        finally:
            for w in writers:w.close()
            if overview is not None:overview.close()
            for view in shared.values():view.renderer.close()
    if not args.preview_only:
        (args.output/'validation.json').write_text(json.dumps(validations,indent=2)+'\n')
        with (args.output/'results.csv').open('w',newline='') as stream:
            fields=['motion','label','seconds','failed','motion_finished','termination_flags']
            w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader()
            w.writerows({**r,'termination_flags':','.join(k for k,v in r['termination_flags'].items() if v)} for r in results)


if __name__=='__main__':main()
