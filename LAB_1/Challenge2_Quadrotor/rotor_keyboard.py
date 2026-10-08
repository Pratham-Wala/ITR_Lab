import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# =========================================================
# MODEL
# =========================================================

# Path is resolved relative to THIS file, so the project works from any
# location / any machine:   LAB_1/models/skydio_x2/scene.xml
XML_PATH = str(
    Path(__file__).resolve().parent.parent
    / "models" / "skydio_x2" / "scene.xml"
)
if not Path(XML_PATH).is_file():
    raise FileNotFoundError(
        f"Model file not found: {XML_PATH}\n"
        "Keep the folder layout from the README (LAB_1/models/skydio_x2/...)."
    )

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


def get_id(obj_type, name):
    """Look up a named object and fail loudly instead of returning -1."""
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"'{name}' not found in {XML_PATH}")
    return idx


# =========================================================
# ACTUATORS
# =========================================================

thrust_motors = [
    get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrust{i}") for i in (1, 2, 3, 4)
]

CTRL_LOW = model.actuator_ctrlrange[thrust_motors, 0]
CTRL_HIGH = model.actuator_ctrlrange[thrust_motors, 1]


# =========================================================
# BODY
# =========================================================

body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, "x2")
print("Body ID:", body_id)


# =========================================================
# START AT TRIMMED HOVER
# =========================================================
# A quadrotor free-falls unless thrust already balances weight.
# The model ships a "hover" keyframe with exactly that trim.

hover_key_id = get_id(mujoco.mjtObj.mjOBJ_KEY, "hover")
mujoco.mj_resetDataKeyframe(model, data, hover_key_id)
HOVER_CTRL = data.ctrl[thrust_motors].copy()
HOVER_MEAN = float(HOVER_CTRL.mean())


# =========================================================
# INITIAL ORIENTATION  (EDIT THESE VALUES)
# =========================================================
# Degrees, intrinsic XYZ euler angles. The body frame is rigidly
# attached to the robot, so this sets the physical attitude AND the
# drawn body-frame triad together.
# NOTE: the controller below levels the drone, so a large initial
# roll/pitch will be corrected right after start-up.

INITIAL_ROLL_DEG = 0.0
INITIAL_PITCH_DEG = 0.0
INITIAL_YAW_DEG = 0.0

INITIAL_POS = np.array([0.0, 0.0, data.qpos[2]])

initial_quat = np.zeros(4)
mujoco.mju_euler2Quat(
    initial_quat,
    np.radians([INITIAL_ROLL_DEG, INITIAL_PITCH_DEG, INITIAL_YAW_DEG]),
    "xyz"
)

data.qpos[0:3] = INITIAL_POS
data.qpos[3:7] = initial_quat
mujoco.mj_forward(model, data)


# =========================================================
# CONTROL  (stabilised "RC-style" flight)
# =========================================================
# The raw X2 has no stability of its own, so the keys do NOT push
# motor outputs directly any more. They set TARGETS:
#
#   W/S : climb-rate target        (m/s)
#   A/D : sideways-speed target    (m/s)   D = move right
#   I/K : forward-speed target     (m/s)   I = forward, K = backward
#   J/L : yaw-rate target          (rad/s) J = turn left (CCW)
#
# Cascaded control: velocity error -> tilt angle target -> motor
# commands (PD). So the drone self-levels, holds altitude AND stops
# horizontally when you let go. Targets relax toward zero (like a
# centred RC stick). Forward/sideways are in the drone's heading
# direction (yaw only), so "forward" stays forward as you turn.

CLIMB_STEP = 0.2         # m/s per keypress
SPEED_STEP = 0.2         # m/s per keypress (forward / sideways)
YAW_RATE_STEP = 0.2      # rad/s per keypress

MAX_CLIMB = 1.5          # m/s
MAX_SPEED = 1.5          # m/s
MAX_TILT = 0.35          # rad (~20 deg), limit on the inner-loop target
MAX_YAW_RATE = 1.5       # rad/s

TARGET_DECAY = 0.995     # per physics step; closer to 1.0 = holds longer

# Controller gains (tuned for the X2 mass/inertia)
KP_ATT, KD_ATT = 2.0, 0.5       # roll / pitch   (angle -> motor units)
KP_VXY = 0.25                   # horizontal speed error -> tilt target (rad per m/s)
KP_YAW = 0.6                    # yaw-rate error -> motor units
KP_VZ = 2.0                     # climb-rate error -> motor units

target_vz = 0.0
target_fwd = 0.0         # forward speed target (heading frame)
target_right = 0.0       # rightward speed target (heading frame)
target_yaw_rate = 0.0


def level_and_hover():
    global target_vz, target_fwd, target_right, target_yaw_rate
    target_vz = target_fwd = target_right = target_yaw_rate = 0.0


# =========================================================
# KEYBOARD
# =========================================================

def key_callback(key):

    global target_vz, target_fwd, target_right, target_yaw_rate

    if key in (ord('W'), ord('w')):
        target_vz += CLIMB_STEP
    elif key in (ord('S'), ord('s')):
        target_vz -= CLIMB_STEP

    elif key in (ord('D'), ord('d')):
        target_right += SPEED_STEP    # move right
    elif key in (ord('A'), ord('a')):
        target_right -= SPEED_STEP    # move left

    elif key in (ord('I'), ord('i')):
        target_fwd += SPEED_STEP      # forward (+x body)
    elif key in (ord('K'), ord('k')):
        target_fwd -= SPEED_STEP      # backward

    elif key in (ord('J'), ord('j')):
        target_yaw_rate += YAW_RATE_STEP   # CCW = left
    elif key in (ord('L'), ord('l')):
        target_yaw_rate -= YAW_RATE_STEP   # CW  = right

    elif key in (ord('X'), ord('x'), 32):  # X or SPACE
        level_and_hover()

    target_vz = float(np.clip(target_vz, -MAX_CLIMB, MAX_CLIMB))
    target_fwd = float(np.clip(target_fwd, -MAX_SPEED, MAX_SPEED))
    target_right = float(np.clip(target_right, -MAX_SPEED, MAX_SPEED))
    target_yaw_rate = float(np.clip(target_yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))


# =========================================================
# CONTROLLER + MIXER
# =========================================================

def body_attitude():
    """Roll, pitch, yaw (rad) from R_WB, ZYX convention.
    +roll  = right-hand rotation about body X (left side goes up)
    +pitch = right-hand rotation about body Y (nose goes DOWN)
    +yaw   = right-hand rotation about body Z (turn LEFT / CCW)"""
    R = data.xmat[body_id].reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw, R


def compute_ctrl():
    """Return the 4 motor commands for the current targets/state.

    Motor layout (from the site positions in x2.xml, body X forward,
    body Y left):
      thrust1: rear-RIGHT  (-0.14, -0.18)   yaw torque sign -
      thrust2: rear-LEFT   (-0.14, +0.18)   yaw torque sign +
      thrust3: front-LEFT  (+0.14, +0.18)   yaw torque sign -
      thrust4: front-RIGHT (+0.14, -0.18)   yaw torque sign +
    Extra thrust on the +y side -> +roll. Extra thrust at the rear
    -> +pitch (nose down). Extra thrust on motors 2 & 4 -> +yaw.
    """
    roll, pitch, yaw, R = body_attitude()
    wx, wy, wz = data.qvel[3:6]          # angular velocity, BODY frame
    vz = data.qvel[2]                    # world vertical velocity

    # Collective: climb-rate loop, tilt-compensated so banking
    # doesn't make the drone sink.
    tilt_comp = 1.0 / max(R[2, 2], 0.5)
    collective = (HOVER_MEAN + KP_VZ * (target_vz - vz)) * tilt_comp - HOVER_MEAN

    # Outer loop: horizontal velocity (in the yaw-aligned heading frame)
    # -> desired tilt. Nose-down (+pitch) accelerates forward; banking
    # right (+roll) accelerates to the right.
    cy, sy = np.cos(yaw), np.sin(yaw)
    v_fwd = cy * data.qvel[0] + sy * data.qvel[1]
    v_right = sy * data.qvel[0] - cy * data.qvel[1]
    target_pitch = np.clip(KP_VXY * (target_fwd - v_fwd), -MAX_TILT, MAX_TILT)
    target_roll = np.clip(KP_VXY * (target_right - v_right), -MAX_TILT, MAX_TILT)

    # Inner loop: attitude PD
    u_roll = KP_ATT * (target_roll - roll) - KD_ATT * wx
    u_pitch = KP_ATT * (target_pitch - pitch) - KD_ATT * wy
    u_yaw = KP_YAW * (target_yaw_rate - wz)

    ctrl = HOVER_CTRL + np.array([
        collective - u_roll + u_pitch - u_yaw,   # thrust1 rear-right
        collective + u_roll + u_pitch + u_yaw,   # thrust2 rear-left
        collective + u_roll - u_pitch - u_yaw,   # thrust3 front-left
        collective - u_roll - u_pitch + u_yaw,   # thrust4 front-right
    ])
    return np.clip(ctrl, CTRL_LOW, CTRL_HIGH)


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
    R = data.xmat[body_id].reshape(3, 3)
    x_B, y_B, z_B = R[:, 0], R[:, 1], R[:, 2]   # body axes in WORLD frame

    axis_length = 0.25

    # ---------------- WORLD (inertial) FRAME -----------------
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

    roll, pitch, yaw, R = body_attitude()
    position = data.xpos[body_id]

    text1 = (
        "SKYDIO X2 QUADROTOR\n\n"
        "Body position [World]:\n"
        f"x = {position[0]: .3f} m\n"
        f"y = {position[1]: .3f} m\n"
        f"z = {position[2]: .3f} m\n\n"
        "Rotation Matrix R_WB:\n"
        f"[ {R[0, 0]: .3f}  {R[0, 1]: .3f}  {R[0, 2]: .3f} ]\n"
        f"[ {R[1, 0]: .3f}  {R[1, 1]: .3f}  {R[1, 2]: .3f} ]\n"
        f"[ {R[2, 0]: .3f}  {R[2, 1]: .3f}  {R[2, 2]: .3f} ]\n\n"
        f"roll {np.degrees(roll): .1f}  pitch {np.degrees(pitch): .1f}  "
        f"yaw {np.degrees(yaw): .1f} deg"
    )

    text2 = (
        "Controls (stabilised):\n"
        "W/S : Climb / Descend\n"
        "A/D : Move left / right\n"
        "I/K : Move forward / back\n"
        "J/L : Yaw left / right\n"
        "X or SPACE : Level + hover\n\n"
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

def physics_step():
    """One controlled physics step (also used by headless tests)."""
    global target_vz, target_fwd, target_right, target_yaw_rate

    data.ctrl[thrust_motors] = compute_ctrl()
    mujoco.mj_step(model, data)

    # Targets relax back to zero -> level + altitude hold when keys
    # are released.
    target_vz *= TARGET_DECAY
    target_fwd *= TARGET_DECAY
    target_right *= TARGET_DECAY
    target_yaw_rate *= TARGET_DECAY


def main():

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback
    ) as viewer:

        viewer.user_scn.ngeom = 0
        next_time = time.perf_counter()

        while viewer.is_running():

            with viewer.lock():
                physics_step()

            draw_frames(viewer)
            update_overlay(viewer)
            viewer.sync()

            next_time += model.opt.timestep
            delay = next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_time = time.perf_counter()


if __name__ == "__main__":
    main()
