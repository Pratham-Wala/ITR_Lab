# ITR Lab 2 - Forward Kinematics with DH Parameters and MuJoCo

Forward kinematics (FK) of two manipulators, computed by hand-written Denavit-Hartenberg (DH)
maths and checked against MuJoCo's own FK:

| Robot | DoF | Script | MuJoCo model |
|-------|-----|--------|--------------|
| Addverb HEAL | 6 | `heal_fk_dh.py` | `robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml` |
| Franka Emika Panda | 7 | `franka_fk_dh.py` | `robot_descriptions/franka/mjx_scene.xml` |

## Lab tasks

1. **Pen & paper** - FK of a 2-DoF manipulator. *(Add the photo/scan here, e.g. `pen_and_paper/2dof_fk.jpg`.)*
2. **Python script** - FK for the 6-DoF HEAL and the 7-DoF Franka, taking a joint vector `q` as input.
3. **MuJoCo** - visualise HEAL and Franka at that joint vector and compare MuJoCo's end-effector pose with the DH result.

## Folder layout

```
ITR_mujoco_fk_lab/
|-- dh_kinematics.py            # maths only: DH tables, dh_transform, forward_kinematics, pose_error (numpy only)
|-- heal_fk_dh.py               # HEAL: load MuJoCo model, set q, compare with DH, open viewer
|-- franka_fk_dh.py             # Franka: same, for the Panda
|-- test_random_poses.py        # checks DH vs MuJoCo on 500 random in-limit poses per robot
|-- requirements.txt / environment.yml
`-- robot_descriptions/         # only the models and meshes these scripts need
```

> `dh_kinematics.py` must stay **all lowercase**. The scripts do `import dh_kinematics`, and a
> capitalised filename fails on Linux.

## Setup

```bash
# pip
pip install -r requirements.txt

# or conda
conda env create -f environment.yml && conda activate itr_fk
```

Only `numpy` and `mujoco` (>= 3.0) are needed.

## Running

Run from inside `ITR_mujoco_fk_lab/`:

```bash
python heal_fk_dh.py                 # prints results, then opens the MuJoCo viewer
python franka_fk_dh.py
python heal_fk_dh.py --no-viewer     # prints results only (no display needed)
python franka_fk_dh.py --no-viewer
python test_random_poses.py          # many-pose sanity check
```

On **macOS** the viewer must be started with `mjpython` instead of `python`
(e.g. `mjpython heal_fk_dh.py`).

### Changing the joint vector

The joint vector is hardcoded at the top of each script, in degrees:

```python
Q_TARGET_DEG = [20.0, -35.0, 50.0, 15.0, -25.0, 40.0]        # heal_fk_dh.py   (q1..q6)
Q_TARGET_DEG = [0.0, -30.0, 0.0, -120.0, 0.0, 90.0, 45.0]    # franka_fk_dh.py (q1..q7)
```

Edit the list and re-run. A warning is printed if any joint is outside its limits.

## Method

Classic (standard) DH convention, metres and radians:

```
T_{i-1,i} = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i),   theta_i = q_i + theta_offset_i
T_0,n     = T_base * T_01 * T_12 * ... * T_{n-1,n}
```

`pose_error` compares the DH end-effector pose with MuJoCo's:
position error = `||p_dh - p_mj||`, orientation error = Frobenius norm of `R_dh - R_mj`.

### Franka (7-DoF)

Franka publishes its parameters in the modified (Craig) convention. `FRANKA_DH_TABLE` is the exact
classic-DH equivalent. MuJoCo's `hand` body is rotated -45 deg about z relative to the flange, so
`FRANKA_HAND_ORIENT_OFFSET` is applied to the orientation check only. `mjx_scene.xml` puts `link0`
at the world origin; if you load `panda.xml` instead, set `T_BASE[:3, 3] = [-0.3, 0, 0.8]`.

### HEAL (6-DoF)

Two tables are provided; `USE_MJCF_EXACT_TABLE` in `heal_fk_dh.py` selects between them.

* **Table A, ideal** (`HEAL_DH_TABLE_IDEAL`): the DH table derived from the robot geometry. Joint 5 is
  tilted by 0.50951 rad, so the wrist is not orthogonal.
* **Table B, MJCF-exact** (`HEAL_DH_TABLE_MJCF`, default): the same chain with the XML's rounding folded
  in (quarter turns written as `1.57` rad, slightly non-normalised quaternions). A few entries such as
  `0.0007963` and `-1.5715927` are therefore **not** geometry; they exist only to reproduce the XML.

## Results

Hardcoded poses (`--no-viewer`):

| Robot | Position error | Orientation error |
|-------|----------------|-------------------|
| Franka | 2.6e-16 m | 4.9e-08 |
| HEAL, table B | 7.0e-08 m | 9.2e-08 |
| HEAL, table A | 4.3e-04 m (0.43 mm) | - |

`test_random_poses.py` (500 random in-limit poses each):

| Robot | Max position error | Max orientation error |
|-------|--------------------|-----------------------|
| Franka | 8.4e-16 m | 4.9e-08 |
| HEAL, table B (MJCF-exact) | 7.9e-08 m | 1.1e-07 |
| HEAL, table A (ideal) | 5.4e-04 m (0.54 mm) | 2.4e-03 |

The DH maths reproduces MuJoCo to numerical precision. For HEAL, table B matches only because it
encodes the XML's rounding; table A shows the true difference between the ideal geometry and the XML.

## Notes / observations

*(Add your own: what you noticed in the viewer, how changing individual joints moves the end effector, etc.)*
