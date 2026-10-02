import time
import mujoco
import mujoco.viewer
import numpy as np

# =========================================================
# MODEL
# =========================================================

XML_PATH = "/home/pratham/mujoco_menagerie/skydio_x2/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# =========================================================
# ACTUATORS
# =========================================================

thrust_motors = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust2"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust3"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust4"),
]

CTRL_LOW = model.actuator_ctrlrange[thrust_motors, 0]
CTRL_HIGH = model.actuator_ctrlrange[thrust_motors, 1]


# =========================================================
# BODY
# =========================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "x2"
)

print("Body ID:", body_id)


# =========================================================
# START AT TRIMMED HOVER
# =========================================================
# Unlike the waffle pi (which just sits on the ground at rest),
# a quadrotor free-falls under gravity unless it starts with
# thrust already balancing its weight. The model ships a "hover"
# keyframe with exactly that trim, so we load it before flying.

hover_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
mujoco.mj_resetDataKeyframe(model, data, hover_key_id)
HOVER_CTRL = data.ctrl[thrust_motors].copy()


# =========================================================
# INITIAL ORIENTATION  (EDIT THESE VALUES)
# =========================================================
# The X2's body frame is rigidly attached to the robot via its
# freejoint (there's no separate offset between "joint frame" and
# "body frame" the way some models have), so there is only ONE
# orientation to set here — it controls the robot's physical
# attitude AND the body-frame triad you see drawn, together.
#
# Angles are in degrees, applied in roll -> pitch -> yaw order
# (intrinsic XYZ euler angles) about the body's own axes.

INITIAL_ROLL_DEG = 0.0     # rotation about body X axis
INITIAL_PITCH_DEG = 0.0    # rotation about body Y axis
INITIAL_YAW_DEG = 0.0      # rotation about body Z axis

# Initial world position (x, y, z) in meters. Leave as-is to keep
# the height set by the "hover" keyframe; override x, y freely.
INITIAL_POS = np.array([0.0, 0.0, data.qpos[2]])

initial_euler_rad = np.radians(
    [INITIAL_ROLL_DEG, INITIAL_PITCH_DEG, INITIAL_YAW_DEG]
)
initial_quat = np.zeros(4)
mujoco.mju_euler2Quat(initial_quat, initial_euler_rad, "xyz")

data.qpos[0:3] = INITIAL_POS
data.qpos[3:7] = initial_quat

# Recompute xpos/xmat (and everything derived) so the very first
# frame drawn and the very first overlay already reflect this
# initial orientation, instead of waiting one mj_step.
mujoco.mj_forward(model, data)


# =========================================================
# CONTROL
# =========================================================

# Smaller steps + hard caps + auto-decay are what actually fix
# "too sensitive" — without a cap, holding a key (which repeats
# key_callback many times per second) lets the trim run away
# unbounded, and without decay it never comes back down on its own.

THRUST_STEP = 0.15     # extra/less lift per keypress (added on top of hover)
ATTITUDE_STEP = 0.05   # roll / pitch / yaw command strength per keypress
# Turn these up if it now feels sluggish, but move in small increments
# (e.g. 0.05 -> 0.08) — this model is very torque-sensitive.

MAX_THRUST_TRIM = 1.5     # hard ceiling on climb/descend trim
MAX_ATTITUDE_TRIM = 0.3   # hard ceiling on roll/pitch/yaw trim

ATTITUDE_DECAY = 0.92  # multiply roll/pitch/yaw trim by this every
                        # physics step -> self-levels when you stop
                        # pressing keys, like a centered RC stick.
                        # Closer to 1.0 = holds attitude longer.
                        # Closer to 0.0 = snaps back to level faster.

thrust_cmd = 0.0
roll_cmd = 0.0
pitch_cmd = 0.0
yaw_cmd = 0.0


# =========================================================
# KEYBOARD
# =========================================================

def key_callback(key):

    global thrust_cmd
    global roll_cmd
    global pitch_cmd
    global yaw_cmd

    if key == ord('W') or key == ord('w'):
        thrust_cmd += THRUST_STEP

    elif key == ord('S') or key == ord('s'):
        thrust_cmd -= THRUST_STEP

    elif key == ord('A') or key == ord('a'):
        roll_cmd -= ATTITUDE_STEP

    elif key == ord('D') or key == ord('d'):
        roll_cmd += ATTITUDE_STEP

    elif key == ord('I') or key == ord('i'):
        pitch_cmd += ATTITUDE_STEP

    elif key == ord('K') or key == ord('k'):
        pitch_cmd -= ATTITUDE_STEP

    elif key == ord('J') or key == ord('j'):
        yaw_cmd -= ATTITUDE_STEP

    elif key == ord('L') or key == ord('l'):
        yaw_cmd += ATTITUDE_STEP

    # Hard-clamp every trim right after the keypress. This is what
    # stops a held key (which fires this callback repeatedly) from
    # accumulating into an ever-larger, harder-to-control command.
    thrust_cmd = float(np.clip(thrust_cmd, -MAX_THRUST_TRIM, MAX_THRUST_TRIM))
    roll_cmd = float(np.clip(roll_cmd, -MAX_ATTITUDE_TRIM, MAX_ATTITUDE_TRIM))
    pitch_cmd = float(np.clip(pitch_cmd, -MAX_ATTITUDE_TRIM, MAX_ATTITUDE_TRIM))
    yaw_cmd = float(np.clip(yaw_cmd, -MAX_ATTITUDE_TRIM, MAX_ATTITUDE_TRIM))

    


def mix_thrust():
    """
    Motor layout (from x2.xml site positions):
      thrust1: rear-left    thrust2: rear-right
      thrust3: front-right  thrust4: front-left
    Sign pattern is a standard X-quad mixer. Result is clipped to
    each motor's ctrlrange (0 to 13 for this model).
    """
    ctrl = np.array([
        HOVER_CTRL[0] + thrust_cmd - roll_cmd - pitch_cmd - yaw_cmd,
        HOVER_CTRL[1] + thrust_cmd + roll_cmd - pitch_cmd + yaw_cmd,
        HOVER_CTRL[2] + thrust_cmd + roll_cmd + pitch_cmd - yaw_cmd,
        HOVER_CTRL[3] + thrust_cmd - roll_cmd + pitch_cmd + yaw_cmd,
    ])
    return np.clip(ctrl, CTRL_LOW, CTRL_HIGH)


# =========================================================
# DRAW BODY FRAME  (unchanged from waffle_keyboard.py)
# =========================================================

def draw_axis(viewer, start, direction, length, rgba):

    start = np.asarray(start, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    end = start + length * direction

    # Make sure there is space for another geometry
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

    # mjv_connector expects two 3-element arrays (from, to),
    # NOT nine separate scalar coordinates.
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.006,
        start,
        end
    )

    viewer.user_scn.ngeom += 1


# =========================================================
# ADD FRAME AXES
# =========================================================

def draw_frames(viewer):

    # Clear previously drawn geometry
    viewer.user_scn.ngeom = 0

    # -----------------------------------------------------
    # Robot position
    # -----------------------------------------------------

    p = data.xpos[body_id].copy()

    # -----------------------------------------------------
    # Rotation matrix
    # -----------------------------------------------------

    R = data.xmat[body_id].reshape(3, 3)

    # Body axes expressed in WORLD frame
    x_B = R[:, 0]
    y_B = R[:, 1]
    z_B = R[:, 2]

    # X2 is small (~0.3m) compared to the waffle pi, so a shorter
    # axis length keeps the triad readable next to the drone.
    axis_length = 0.25

    # -----------------------------------------------------
    # WORLD / INERTIAL FRAME
    # -----------------------------------------------------
    # NOTE: as in the original waffle script, this draws the WORLD
    # frame at a fixed offset point, labeled "inertial" for contrast
    # with the moving body frame. If you want the drone's true
    # inertial (center-of-mass / principal-axis) frame instead, swap
    # p -> data.xipos[body_id] and R -> data.ximat[body_id].reshape(3,3)
    # in a second draw_frame call below.

    world_origin = np.array([-0.25, -0.25, 0.0])
    world_axis_length = 0.4

    # X_W
    draw_axis(
        viewer,
        world_origin,
        np.array([1.0, 0.0, 0.0]),
        world_axis_length,
        np.array([0.9, 0.55, 0.1, 1.0])   # orange
    )

    # Y_W
    draw_axis(
        viewer,
        world_origin,
        np.array([0.0, 1.0, 0.0]),
        world_axis_length,
        np.array([0.1, 0.8, 0.8, 1.0])    # cyan
    )

    # Z_W
    draw_axis(
        viewer,
        world_origin,
        np.array([0.0, 0.0, 1.0]),
        world_axis_length,
        np.array([0.6, 0.2, 0.9, 1.0])    # purple
    )

    # -----------------------------------------------------
    # BODY FRAME
    # -----------------------------------------------------

    # X_B
    draw_axis(
        viewer,
        p,
        x_B,
        axis_length,
        np.array([1.0, 0.0, 0.0, 1.0])
    )

    # Y_B
    draw_axis(
        viewer,
        p,
        y_B,
        axis_length,
        np.array([0.0, 1.0, 0.0, 1.0])
    )

    # Z_B
    draw_axis(
        viewer,
        p,
        z_B,
        axis_length,
        np.array([0.0, 0.0, 1.0, 1.0])
    )


# =========================================================
# ROTATION MATRIX TEXT
# =========================================================

def update_overlay(viewer):

    R = data.xmat[body_id].reshape(3, 3)
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
        f"[ {R[2, 0]: .3f}  {R[2, 1]: .3f}  {R[2, 2]: .3f} ]"
    )

    text2 = (
        "Controls:\n"
        "W/S : Thrust up / down\n"
        "A/D : Roll left / right\n"
        "I/K : Pitch forward / back\n"
        "J/L : Yaw left / right\n"
        "SPACE : Level trims (hover)"
    )

    # The passive viewer has no `.overlays` dict — text overlays
    # are set via set_texts((font, gridpos, text1, text2)).
    viewer.set_texts((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        text1,
        text2
    ))


# =========================================================
# VIEWER
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    # Reset user geometry counter
    viewer.user_scn.ngeom = 0

    while viewer.is_running():

        # Rotor thrust mixing
        data.ctrl[thrust_motors] = mix_thrust()

        # Physics
        mujoco.mj_step(model, data)

        # Auto-level: relax roll/pitch/yaw trims back toward zero
        # every step. Thrust is left alone (you want climb/descend
        # to persist), but attitude decaying is what makes the
        # drone feel controllable instead of twitchy.
        roll_cmd *= ATTITUDE_DECAY
        pitch_cmd *= ATTITUDE_DECAY
        yaw_cmd *= ATTITUDE_DECAY

        # Draw body + world frame axes
        draw_frames(viewer)

        # Display rotation matrix / position / controls
        update_overlay(viewer)

        # Update viewer
        viewer.sync()

        time.sleep(model.opt.timestep)