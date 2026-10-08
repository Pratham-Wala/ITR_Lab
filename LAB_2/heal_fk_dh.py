#!/usr/bin/env python3
"""
HEAL robot - MuJoCo side ONLY (load model, set joints, read EE pose, viewer).
The DH maths and the error calculation live in dh_kinematics.py.

Keep dh_kinematics.py (all lowercase) in the same folder, next to robot_descriptions/.
Run from inside ITR_mujoco_fk_lab/:
    pip install -r requirements.txt
    python heal_fk_dh.py              # (macOS: mjpython heal_fk_dh.py)
    python heal_fk_dh.py --no-viewer  # print results only
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import dh_kinematics as dhk          # <-- the maths file

# =============================================================================
# USER INPUT
# =============================================================================
LAB_ROOT = Path(__file__).resolve().parent
MODEL_PATH = LAB_ROOT / "robot_descriptions" / "single_arm_heal_effort_actuation_rs_mj.xml"

# Target joint angles q1..q6 in DEGREES
Q_TARGET_DEG = [20.0, -35.0, 50.0, 15.0, -25.0, 40.0]

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
EE_BODY_NAME = "end_effector"

# True  -> DH table that reproduces the MJCF's rounded 1.57 rad values (error ~1e-7 m)
# False -> ideal exact-pi/2 table (error ~0.5 mm vs the MJCF)
USE_MJCF_EXACT_TABLE = True
DH_TABLE = dhk.HEAL_DH_TABLE_MJCF if USE_MJCF_EXACT_TABLE else dhk.HEAL_DH_TABLE_IDEAL

# base_link is at the world origin (no rotation) in this MJCF
T_BASE = np.eye(4)


def set_joint_angles(model, data, q):
    for name, qi in zip(JOINT_NAMES, q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in the MJCF.")
        data.qpos[model.jnt_qposadr[jid]] = qi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true", help="print results and exit")
    args = ap.parse_args()

    q = np.deg2rad(np.array(Q_TARGET_DEG, dtype=float))

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    for name, qi in zip(JOINT_NAMES, q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        if model.jnt_limited[jid] and not (lo <= qi <= hi):
            print(f"[warn] {name} = {np.rad2deg(qi):.1f} deg is outside its limits "
                  f"[{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}] deg")

    # ---- MuJoCo forward kinematics ------------------------------------------
    set_joint_angles(model, data, q)
    mujoco.mj_kinematics(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    if bid < 0:
        raise ValueError(f"Body '{EE_BODY_NAME}' not found in the MJCF.")
    p_mj = data.xpos[bid].copy()
    R_mj = data.xmat[bid].reshape(3, 3).copy()

    # ---- DH maths + error (from dh_kinematics.py) ---------------------------
    T_dh = dhk.forward_kinematics(q, DH_TABLE, T_base=T_BASE)
    res = dhk.pose_error(T_dh, p_mj, R_mj)

    p_dh, e = res["p_dh"], res["err_vec"]
    print(f"DH table used: {'MJCF-exact (B)' if USE_MJCF_EXACT_TABLE else 'ideal (A)'}")
    print(f"MuJoCo   EE position X, Y, Z = {p_mj[0]:.6f}, {p_mj[1]:.6f}, {p_mj[2]:.6f} m")
    print(f"DH maths EE position X, Y, Z = {p_dh[0]:.6f}, {p_dh[1]:.6f}, {p_dh[2]:.6f} m")
    print(f"Error (DH - MuJoCo)  dX, dY, dZ = {e[0]:.3e}, {e[1]:.3e}, {e[2]:.3e} m")
    print(f"Position error norm  : {res['pos_err']:.3e} m ({res['pos_err'] * 1000:.3e} mm)")
    print(f"Orientation error    : {res['rot_err']:.3e} (Frobenius norm of R_dh - R_mj)")

    if args.no_viewer:
        return

    # ---- Passive viewer -----------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                set_joint_angles(model, data, q)
                mujoco.mj_kinematics(model, data)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()