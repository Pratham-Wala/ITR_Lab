import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# =========================================================
# MODEL
# =========================================================

# Path is resolved relative to THIS file, so the project works from any
# location / any machine:   LAB_1/models/robotis_tb3/scene_...xml
XML_PATH = str(
    Path(__file__).resolve().parent.parent
    / "models" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"
)
if not Path(XML_PATH).is_file():
    raise FileNotFoundError(
        f"Model file not found: {XML_PATH}\n"
        "Keep the folder layout from the README (LAB_1/models/robotis_tb3/...)."
    )

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


def get_id(obj_type, name):
    """Look up a named object and fail loudly instead of returning -1
    (a -1 would silently index the LAST actuator/body in numpy)."""
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"'{name}' not found in {XML_PATH}")
    return idx


# =========================================================
# ACTUATORS
# =========================================================

left_motor = get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
right_motor = get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")


# =========================================================
# BODY
# =========================================================

body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, "base")
print("Body ID:", body_id)


# =========================================================
# INITIAL ORIENTATION
# =========================================================
# Starting roll/pitch/yaw in degrees. For a ground robot you
# normally only change YAW. This is the robot's REAL orientation,
# and the drawn body frame + printed matrix both follow it.

INITIAL_ROLL_DEG = 0.0
INITIAL_PITCH_DEG = 0.0
INITIAL_YAW_DEG = 90.0

joint_id = model.body_jntadr[body_id]
qpos_adr = model.jnt_qposadr[joint_id]

if model.body_jntnum[body_id] > 0 and \
        model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:

    euler = np.deg2rad([INITIAL_ROLL_DEG, INITIAL_PITCH_DEG, INITIAL_YAW_DEG])
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, euler, "xyz")

    # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
    data.qpos[qpos_adr + 3: qpos_adr + 7] = quat
    mujoco.mj_forward(model, data)
else:
    print("WARNING: 'base' body has no freejoint - "
          "initial orientation was not applied.")


# =========================================================
# CONTROL
# =========================================================
# Wheel actuators are velocity servos limited to +/-7.88 rad/s,
# so keep the commands inside that range.

WHEEL_LIMIT = float(model.actuator_ctrlrange[left_motor, 1])
FORWARD_SPEED = min(5.0, WHEEL_LIMIT)
TURN_SPEED = min(7.0, WHEEL_LIMIT)

left_velocity = 0.0
right_velocity = 0.0


# =========================================================
# KEYBOARD
# =========================================================

def key_callback(key):

    global left_velocity
    global right_velocity

    if key in (ord('W'), ord('w')):
        left_velocity = FORWARD_SPEED
        right_velocity = FORWARD_SPEED

    elif key in (ord('S'), ord('s')):
        left_velocity = -FORWARD_SPEED
        right_velocity = -FORWARD_SPEED

    elif key in (ord('A'), ord('a')):
        left_velocity = -TURN_SPEED
        right_velocity = TURN_SPEED

    elif key in (ord('D'), ord('d')):
        left_velocity = TURN_SPEED
        right_velocity = -TURN_SPEED

    # X is the reliable stop key. SPACE also works, but the MuJoCo
    # viewer uses SPACE for its own pause toggle.
    elif key in (ord('X'), ord('x'), 32):
        left_velocity = 0.0
        right_velocity = 0.0


# =========================================================
# DRAW FRAMES
# =========================================================

def draw_axis(viewer, start, direction, length, rgba):

    start = np.asarray(start, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    end = start + length * direction

    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        rgba
    )

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.006,
        start,
        end
    )

    viewer.user_scn.ngeom += 1


def draw_frames(viewer):

    viewer.user_scn.ngeom = 0

    p = data.xpos[body_id].copy()

    # Body axes expressed in the WORLD frame = columns of R_WB.
    # No display offset: these arrows ARE the robot's real frame and
    # match the matrix printed in the overlay exactly.
    R = data.xmat[body_id].reshape(3, 3)
    x_B, y_B, z_B = R[:, 0], R[:, 1], R[:, 2]

    axis_length = 0.4

    # ---------------- WORLD (inertial) FRAME -----------------
    # Drawn at a fixed point beside the origin so it does not sit
    # on top of the robot's own frame at start-up.
    world_origin = np.array([-0.25, -0.25, 0.0])
    world_axis_length = 0.4

    draw_axis(viewer, world_origin, [1, 0, 0], world_axis_length, np.array([0.9, 0.55, 0.1, 1.0]))  # orange
    draw_axis(viewer, world_origin, [0, 1, 0], world_axis_length, np.array([0.1, 0.8, 0.8, 1.0]))   # cyan
    draw_axis(viewer, world_origin, [0, 0, 1], world_axis_length, np.array([0.6, 0.2, 0.9, 1.0]))   # purple

    # ---------------- BODY FRAME -----------------------------
    draw_axis(viewer, p, x_B, axis_length, np.array([1.0, 0.0, 0.0, 1.0]))  # red
    draw_axis(viewer, p, y_B, axis_length, np.array([0.0, 1.0, 0.0, 1.0]))  # green
    draw_axis(viewer, p, z_B, axis_length, np.array([0.0, 0.0, 1.0, 1.0]))  # blue


# =========================================================
# OVERLAY TEXT
# =========================================================

def update_overlay(viewer):

    R = data.xmat[body_id].reshape(3, 3)
    position = data.xpos[body_id]

    # ZYX (yaw-pitch-roll) angles recovered from R, for a quick sanity
    # check of the matrix.
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

    text1 = (
        "TURTLEBOT3 WAFFLE PI\n\n"
        "Body position [World]:\n"
        f"x = {position[0]: .3f} m\n"
        f"y = {position[1]: .3f} m\n"
        f"z = {position[2]: .3f} m\n\n"
        "Rotation Matrix R_WB:\n"
        f"[ {R[0, 0]: .3f}  {R[0, 1]: .3f}  {R[0, 2]: .3f} ]\n"
        f"[ {R[1, 0]: .3f}  {R[1, 1]: .3f}  {R[1, 2]: .3f} ]\n"
        f"[ {R[2, 0]: .3f}  {R[2, 1]: .3f}  {R[2, 2]: .3f} ]\n\n"
        f"roll {roll: .1f}  pitch {pitch: .1f}  yaw {yaw: .1f} deg"
    )

    text2 = (
        "Controls:\n"
        "W/S : Forward / Backward\n"
        "A/D : Rotate Left / Right\n"
        "X or SPACE : Stop\n\n"
        "Frames:\n"
        "BODY  : X red, Y green, Z blue\n"
        "WORLD : X orange, Y cyan, Z purple\n"
        "        (drawn at -0.25, -0.25, 0)"
    )

    viewer.set_texts((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        text1,
        text2
    ))


# =========================================================
# VIEWER
# =========================================================

def main():

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback
    ) as viewer:

        viewer.user_scn.ngeom = 0
        next_time = time.perf_counter()

        while viewer.is_running():

            # Physics (locked so the render thread never reads a
            # half-updated state)
            with viewer.lock():
                data.ctrl[left_motor] = left_velocity
                data.ctrl[right_motor] = right_velocity
                mujoco.mj_step(model, data)

            draw_frames(viewer)
            update_overlay(viewer)
            viewer.sync()

            # Real-time pacing that accounts for compute time
            next_time += model.opt.timestep
            delay = next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = time.perf_counter()


if __name__ == "__main__":
    main()
