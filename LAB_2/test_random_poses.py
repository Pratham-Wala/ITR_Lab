#!/usr/bin/env python3
"""
Sanity check: compare the DH forward kinematics with MuJoCo on many random,
in-limit joint configurations (not just the single pose hardcoded in the main scripts).

    python test_random_poses.py
"""
from pathlib import Path

import numpy as np
import mujoco

import dh_kinematics as dhk

ROOT = Path(__file__).resolve().parent / "robot_descriptions"
N_POSES = 500
rng = np.random.default_rng(0)

ROBOTS = [
    # name, xml, joint names, ee body, DH table, orientation offset
    ("Franka (7-DoF)", ROOT / "franka" / "mjx_scene.xml",
     [f"joint{i}" for i in range(1, 8)], "hand",
     dhk.FRANKA_DH_TABLE, dhk.FRANKA_HAND_ORIENT_OFFSET),
    ("HEAL (6-DoF), MJCF-exact table", ROOT / "single_arm_heal_effort_actuation_rs_mj.xml",
     [f"joint_{i}" for i in range(1, 7)], "end_effector",
     dhk.HEAL_DH_TABLE_MJCF, None),
    ("HEAL (6-DoF), ideal table", ROOT / "single_arm_heal_effort_actuation_rs_mj.xml",
     [f"joint_{i}" for i in range(1, 7)], "end_effector",
     dhk.HEAL_DH_TABLE_IDEAL, None),
]


def main():
    for name, xml, joints, ee, table, orient in ROBOTS:
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]

        pos_errs, rot_errs = [], []
        for _ in range(N_POSES):
            q = np.array([rng.uniform(*model.jnt_range[j]) for j in jids])
            for j, qi in zip(jids, q):
                data.qpos[model.jnt_qposadr[j]] = qi
            mujoco.mj_kinematics(model, data)
            T = dhk.forward_kinematics(q, table)
            r = dhk.pose_error(T, data.xpos[bid], data.xmat[bid].reshape(3, 3),
                               T_orient_offset=orient)
            pos_errs.append(r["pos_err"])
            rot_errs.append(r["rot_err"])

        print(f"{name:34s} {N_POSES} poses | max pos err = {max(pos_errs):.2e} m "
              f"| max rot err = {max(rot_errs):.2e}")


if __name__ == "__main__":
    main()
