"""
dh_kinematics.py - the MATHS ONLY module (numpy only, no MuJoCo, no quaternions).

Contains
  * the classic Denavit-Hartenberg tables for the Franka Panda and the HEAL robot
  * dh_transform / forward_kinematics  (4x4 homogeneous matrices)
  * pose_error                          (DH result vs. a reference pose)

Convention (classic / standard DH), units: metres and radians
    T_{i-1,i} = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
    theta_i   = q_i + theta_offset_i
"""
import numpy as np


# =============================================================================
# Core maths
# =============================================================================
def rot_z(a):
    """4x4 homogeneous rotation about z."""
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def dh_transform(theta, d, a, alpha):
    """Classic DH homogeneous transform for one joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def forward_kinematics(q, dh_table, T_base=None, T_tool=None):
    """
    q        : joint angles [rad], one per DH row
    dh_table : list of (theta_offset, d, a, alpha)
    T_base   : fixed 4x4 world -> DH frame 0   (default identity)
    T_tool   : fixed 4x4 last DH frame -> tool (default identity)
    returns  : 4x4 pose of the tool frame in the world frame
    """
    T = np.eye(4) if T_base is None else T_base.copy()
    for qi, (theta_off, d, a, alpha) in zip(q, dh_table):
        T = T @ dh_transform(qi + theta_off, d, a, alpha)
    return T if T_tool is None else T @ T_tool


def pose_error(T_dh, p_ref, R_ref, T_orient_offset=None):
    """
    Compare the DH pose with a reference position p_ref (3,) and rotation matrix R_ref (3x3).
    T_orient_offset: optional fixed 4x4 applied to the DH pose before comparing orientation
                     (e.g. a frame that differs from the DH frame by a fixed rotation).
    """
    p_dh = T_dh[:3, 3]
    T_cmp = T_dh if T_orient_offset is None else T_dh @ T_orient_offset
    err_vec = p_dh - np.asarray(p_ref)
    return {
        "p_dh": p_dh,
        "err_vec": err_vec,
        "pos_err": float(np.linalg.norm(err_vec)),
        "rot_err": float(np.linalg.norm(T_cmp[:3, :3] - np.asarray(R_ref))),
    }


# =============================================================================
# FRANKA EMIKA PANDA - classic DH table
#
#   joint | theta_offset |   d      |    a     |  alpha
#   ------+--------------+----------+----------+---------
#     1   |      0       |  0.333   |   0      | -pi/2
#     2   |      0       |  0       |   0      | +pi/2
#     3   |      0       |  0.316   |   0.0825 | +pi/2
#     4   |      0       |  0       |  -0.0825 | -pi/2
#     5   |      0       |  0.384   |   0      | +pi/2
#     6   |      0       |  0       |   0.088  | +pi/2
#     7   |      0       |  0.107   |   0      |  0        <- last frame = flange
#
# (Franka's published table is the modified/Craig convention; these numbers are
#  its exact classic-DH equivalent.)
# =============================================================================
FRANKA_DH_TABLE = [
    # theta_offset,  d,      a,       alpha
    (0.0,           0.333,   0.0,     -np.pi / 2),
    (0.0,           0.0,     0.0,      np.pi / 2),
    (0.0,           0.316,   0.0825,   np.pi / 2),
    (0.0,           0.0,    -0.0825,  -np.pi / 2),
    (0.0,           0.384,   0.0,      np.pi / 2),
    (0.0,           0.0,     0.088,    np.pi / 2),
    (0.0,           0.107,   0.0,      0.0),
]

# MuJoCo's "hand" body is rotated -45 deg about z w.r.t. the flange (orientation only).
FRANKA_HAND_ORIENT_OFFSET = rot_z(-np.pi / 4)


# =============================================================================
# HEAL (Addverb, 6-DOF) - classic DH tables
# Frame z-axes follow the MuJoCo joint axes at q=0, so q_i maps to theta_i with no
# sign flips (joint_2 / joint_6 have axis "0 0 -1" in the MJCF; already absorbed).
#
# --- Table A: IDEAL geometry (exact pi/2's) ---------------------------------
#   joint | theta_offset |   d       |  a   |  alpha
#   ------+--------------+-----------+------+---------
#     1   |     pi       |  0.3208   |  0   |  pi/2      base 0.171 + 0.1498
#     2   |     pi/2     |  0        |  0.3 |  pi        upper arm 0.3 m
#     3   |     0        |  0        |  0   |  pi/2      elbow
#     4   |     pi       |  0.37736  |  0   |  0.50951   wrist is NOT orthogonal:
#     5   |    -pi/2     |  0        |  0   |  pi/2      joint 5 tilted 0.50951 rad
#     6   |     pi       | -0.1227   |  0   |  pi        tool flange, z pointing down
#   (d4 = 0.1593 + 0.16105 + 0.03185/tan(0.50951))
#
# --- Table B: MJCF-exact ------------------------------------------------------
# The HEAL MJCF writes quarter turns as 1.57 rad (not pi/2) and uses slightly
# non-normalised quaternions. Table B is the same chain with those rounded values
# folded in, so it reproduces MuJoCo to ~1e-7 m. Table A differs by up to ~0.5 mm.
# =============================================================================
HEAL_DH_TABLE_IDEAL = [
    # theta_offset,  d,        a,    alpha
    (np.pi,          0.3208,   0.0,  np.pi / 2),
    (np.pi / 2,      0.0,      0.3,  np.pi),
    (0.0,            0.0,      0.0,  np.pi / 2),
    (np.pi,          0.37736,  0.0,  0.50951),
    (-np.pi / 2,     0.0,      0.0,  np.pi / 2),
    (np.pi,         -0.1227,   0.0,  np.pi),
]

HEAL_DH_TABLE_MJCF = [
    # theta_offset,  d,          a,    alpha
    (np.pi,          0.3207996,  0.0,  1.5707921),
    (np.pi / 2,      0.0,        0.3,  np.pi),
    (0.0007963,     -0.0001269,  0.0,  1.57),
    (np.pi,          0.3773558,  0.0,  0.50951),
    (-1.5715927,     0.0001006,  0.0,  1.5707921),
    (np.pi,         -0.1227,     0.0,  np.pi),
]