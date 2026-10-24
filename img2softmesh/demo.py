"""Scripted grasp demo of a generated scene (interactive viewer or mp4): lift, shake, lay back down flat."""

import time

import cv2
import mujoco
import numpy as np


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


class GraspScript:
    """settle -> grab -> lift until it hangs free -> shake -> carry over the cloth -> lay it down flat -> release.

    All grippers move together, holding their vertex patches. Assumes a generated scene: mocap body g is
    gripper<g> and every connect equality is a grasp<g>_* (the flex's own edge equality stays on).
    """

    def __init__(self, model):
        self.m = model
        self.eq = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_CONNECT)
        self.vbody = [model.equality(f"grasp{g}_0").obj1id[0] for g in range(model.nmocap)]  # vertex under each
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        xyz = d.flexvert_xpos
        self.span = np.ptp(xyz[:, :2], axis=0).max()  # cloth size, not the whole scene
        self.z0 = xyz[:, 2].min()
        self.hang = d.xpos[self.vbody, 1].mean() - xyz[:, 1].min()  # length that hangs below the grippers
        self.lift = 1.15 * self.hang + 0.05
        phases = {"settle": 0.6, "lift": 1.4, "shake": 2.0, "carry": 1.0, "lay": 2.0, "release": 0.1, "rest": 1.4}
        self.names, self.durs = list(phases), np.array(list(phases.values()))
        self.ends = np.cumsum(self.durs)
        self.duration = self.ends[-1]

    def reset(self, d):
        mujoco.mj_resetData(self.m, d)
        d.eq_active[self.eq] = 0
        self.start = None

    def offset(self, name, s):
        """Common gripper displacement from the grab position in phase `name` at eased progress s."""
        back = -0.95 * self.hang  # over the far edge: laying down from there puts the cloth back in place
        if name == "lift":
            return [0, 0, self.lift * s]
        if name == "shake":  # two swings across the hanging cloth, fading in and out
            return [0, 0.12 * self.hang * np.sin(4 * np.pi * s) * np.sin(np.pi * s), self.lift]
        touch = self.hang + 0.01  # gripper height at which the hanging hem just touches the table
        if name == "carry":
            return [0, back * s, self.lift + (touch - self.lift) * s]
        if name == "lay":  # travel back while lowering at the same rate: the cloth unrolls flat onto the table
            return [0, back * (1 - s), 0.01 + (touch - 0.01) * (1 - s)]
        return None

    def step(self, d):
        i = min(np.searchsorted(self.ends, d.time, side="right"), len(self.ends) - 1)
        name = self.names[i]
        tau = (d.time - (self.ends[i] - self.durs[i])) / self.durs[i]
        if name == "settle":
            d.mocap_pos[:] = d.xpos[self.vbody]
            return
        if self.start is None:  # grab now, at the settled positions
            self.start = d.xpos[self.vbody].copy()
            d.eq_active[self.eq] = 1
        off = self.offset(name, np.clip(tau, 0, 1) if name == "shake" else _smooth(tau))
        if off is None:
            d.eq_active[self.eq] = 0
        else:
            d.mocap_pos[:] = self.start + off


def _load(scene):
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    script = GraspScript(m)
    script.reset(d)
    return m, d, script


def _advance(m, d, script, dt):
    for _ in range(max(1, round(dt / m.opt.timestep))):
        script.step(d)
        mujoco.mj_step(m, d)


def view(scene, loop=True):
    import mujoco.viewer
    m, d, script = _load(scene)
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.perf_counter()
            if d.time > script.duration:
                if not loop:
                    break
                script.reset(d)
            _advance(m, d, script, 1 / 60)
            v.sync()
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - t0)))


def record(scene, path, width=960, height=720, fps=30):
    m, d, script = _load(scene)
    cam = mujoco.MjvCamera()  # framed on the cloth and its lift, not the whole scene
    cam.lookat[:] = [0, -0.3 * script.hang, script.z0 + 0.45 * script.lift]
    cam.distance = 1.7 * max(script.span, script.lift)
    cam.azimuth, cam.elevation = 125, -22
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    t0 = time.perf_counter()
    with mujoco.Renderer(m, height, width) as r:
        while d.time < script.duration:
            _advance(m, d, script, 1 / fps)
            r.update_scene(d, cam)
            writer.write(r.render()[:, :, ::-1])
    writer.release()
    if not np.isfinite(d.flexvert_xpos).all():
        raise RuntimeError("simulation diverged (NaN); try a smaller timestep or fewer vertices")
    return d.time / (time.perf_counter() - t0)
