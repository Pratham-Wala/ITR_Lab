import time
import mujoco
import mujoco.viewer
import numpy as np

# =========================================================
# MODEL
# =========================================================

XML_PATH = "/home/pratham/robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# =========================================================
# ACTUATORS
# =========================================================

left_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_left"
)

right_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_right"
)


# =========================================================
# BODY
# =========================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)

print("Body ID:", body_id)


# =========================================================
# INITIAL ORIENTATION
# =========================================================
# Set the robot's starting yaw/pitch/roll (in degrees) here.
# For a ground robot you'll usually only want to change YAW.

INITIAL_ROLL_DEG = 0.0
INITIAL_PITCH_DEG = 0.0
INITIAL_YAW_DEG = 90.0

joint_id = model.body_jntadr[body_id]
qpos_adr = model.jnt_qposadr[joint_id]

# body_jntnum tells us if this body has a joint at all, and
# jnt_type tells us if it's a freejoint (7 qpos: xyz + quat)
if model.body_jntnum[body_id] > 0 and \
        model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:

    euler = np.array([
        np.deg2rad(INITIAL_ROLL_DEG),
        np.deg2rad(INITIAL_PITCH_DEG),
        np.deg2rad(INITIAL_YAW_DEG)
    ])

    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, euler, "xyz")

    # qpos layout for a freejoint: [x, y, z, qw, qx, qy, qz]
    data.qpos[qpos_adr + 3: qpos_adr + 7] = quat

    # Recompute derived quantities (xpos, xmat, etc.) so the
    # change is reflected immediately, before the first mj_step
    mujoco.mj_forward(model, data)
else:
    print(
        "WARNING: 'base' body has no freejoint — "
        "initial orientation was not applied."
    )


# =========================================================
# CONTROL
# =========================================================

FORWARD_SPEED = 5.0
TURN_SPEED = 10.0

left_velocity = 0.0
right_velocity = 0.0


# =========================================================
# BODY FRAME DISPLAY OFFSET
# =========================================================
# This rotates ONLY the drawn body-frame axes relative to the
# robot's actual physical orientation (e.g. if you want the
# reference frame to represent a rotated sensor/mount frame
# instead of exactly matching the mesh). Leave all at 0 if you
# just want the axes to match the robot's real orientation.

BODY_FRAME_ROLL_DEG = 0.0
BODY_FRAME_PITCH_DEG = 0.0
BODY_FRAME_YAW_DEG = 45.0

_body_frame_offset_euler = np.array([
    np.deg2rad(BODY_FRAME_ROLL_DEG),
    np.deg2rad(BODY_FRAME_PITCH_DEG),
    np.deg2rad(BODY_FRAME_YAW_DEG)
])

_body_frame_offset_quat = np.zeros(4)
mujoco.mju_euler2Quat(_body_frame_offset_quat, _body_frame_offset_euler, "xyz")

BODY_FRAME_OFFSET_R = np.zeros(9)
mujoco.mju_quat2Mat(BODY_FRAME_OFFSET_R, _body_frame_offset_quat)
BODY_FRAME_OFFSET_R = BODY_FRAME_OFFSET_R.reshape(3, 3)


# =========================================================
# KEYBOARD
# =========================================================

def key_callback(key):

    global left_velocity
    global right_velocity

    if key == ord('W') or key == ord('w'):
        left_velocity = FORWARD_SPEED
        right_velocity = FORWARD_SPEED

    elif key == ord('S') or key == ord('s'):
        left_velocity = -FORWARD_SPEED
        right_velocity = -FORWARD_SPEED

    elif key == ord('A') or key == ord('a'):
        left_velocity = -TURN_SPEED
        right_velocity = TURN_SPEED

    elif key == ord('D') or key == ord('d'):
        left_velocity = TURN_SPEED
        right_velocity = -TURN_SPEED

    elif key == 32:  # SPACE
        left_velocity = 0.0
        right_velocity = 0.0


# =========================================================
# DRAW BODY FRAME
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

    # Apply the display-only offset rotation (expressed in the
    # body frame) so the drawn axes can differ from the robot's
    # exact physical orientation if BODY_FRAME_*_DEG != 0.
    R_display = R @ BODY_FRAME_OFFSET_R

    # Body axes expressed in WORLD frame
    x_B = R_display[:, 0]
    y_B = R_display[:, 1]
    z_B = R_display[:, 2]

    axis_length = 0.4

    # -----------------------------------------------------
    # WORLD / INERTIAL FRAME
    # -----------------------------------------------------

    world_origin = np.array([-0.25, -0.25, 0.0])
    world_axis_length = 0.4

    # World frame drawn in muted/pastel colors so it's visually
    # distinct from the body frame (which uses bright primaries).
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
        "TURTLEBOT3 WAFFLE PI\n\n"
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
        "W/S : Forward / Backward\n"
        "A/D : Rotate Left / Right\n"
        "SPACE : Stop"
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

        # Wheel control
        data.ctrl[left_motor] = left_velocity
        data.ctrl[right_motor] = right_velocity

        # Physics
        mujoco.mj_step(model, data)

        # Draw body + world frame axes
        draw_frames(viewer)

        # Display rotation matrix / position / controls
        update_overlay(viewer)

        # Update viewer
        viewer.sync()

        time.sleep(model.opt.timestep)