# LAB 1 – Rigid-Body Frames & Rotation Matrices in MuJoCo

Two keyboard-driven MuJoCo simulations that show how a robot's **body frame**
moves relative to the **world frame**, together with the live **rotation
matrix R_WB**.

| Challenge | Robot | Script |
|-----------|-------|--------|
| 1 | TurtleBot3 Waffle Pi (ground robot) | `Challenge1_TurtleBot/waffle_keyboard.py` |
| 2 | Skydio X2 (quadrotor) | `Challenge2_Quadrotor/rotor_keyboard.py` |

## Requirements covered

| # | Task from the lab | Where |
|---|-------------------|-------|
| C1-1 | Spawn TurtleBot Waffle Pi | `waffle_keyboard.py` (loads `models/robotis_tb3`) |
| C1-2 | Use keyboard to move | `key_callback()` in the script |
| C1-3 | Show how the body frame changes | Coloured arrows drawn on the robot every frame (`draw_frames()`) |
| C1-4 | Display rotation matrix | On-screen overlay, top-left (`update_overlay()`) |
| C2-1 | Spawn any quadrotor | `rotor_keyboard.py` (loads `models/skydio_x2`) |
| C2-2/3/4 | Repeat keyboard, body frame, rotation matrix | Same three features, implemented for the quadrotor |

## Folder layout (keep it as is)

```
LAB_1/
├── README.md
├── requirements.txt
├── Challenge1_TurtleBot/
│   └── waffle_keyboard.py
├── Challenge2_Quadrotor/
│   └── rotor_keyboard.py
└── models/                      <- model files bundled with the project
    ├── robotis_tb3/             (TurtleBot3 Waffle Pi, Apache-2.0)
    └── skydio_x2/               (Skydio X2, Apache-2.0)
```

The scripts find the models **relative to their own location**
(`Path(__file__)`), so no paths need editing and the folder can live anywhere.
If a model is missing you get a clear `FileNotFoundError` naming the path.

## Setup

Python 3.9+ is needed.

```bash
cd LAB_1
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python Challenge1_TurtleBot/waffle_keyboard.py
python Challenge2_Quadrotor/rotor_keyboard.py
```

> **macOS:** the passive viewer must be started with `mjpython` instead of
> `python` (e.g. `mjpython Challenge1_TurtleBot/waffle_keyboard.py`).
> **Linux over SSH / no display:** the viewer needs a display (or X forwarding).

Click the MuJoCo window first so it receives the key presses.

## Controls

### Challenge 1 – TurtleBot3 Waffle Pi

| Key | Action |
|-----|--------|
| W / S | Forward / backward |
| A / D | Rotate left / right (on the spot) |
| X or SPACE | Stop |

Commands latch: the robot keeps moving until you press X / SPACE.
(SPACE is also the viewer's own pause key, so X is the reliable stop.)

### Challenge 2 – Skydio X2 quadrotor

| Key | Action |
|-----|--------|
| W / S | Climb / descend |
| I / K | Move forward / backward |
| A / D | Move left / right |
| J / L | Yaw left / right |
| X or SPACE | Zero all targets: level + hover |

Each press adds a small step; holding a key accumulates up to a limit. When
you release the keys the targets relax to zero, so the drone levels out,
holds its altitude and stops.

**Why a controller?** The raw X2 has no stability of its own: pushing motor
outputs directly makes it tumble within seconds. The keys therefore set
*targets* (climb rate, horizontal speed, yaw rate) and a cascaded PD
controller turns them into the four motor commands:
velocity error → tilt target → attitude PD → motor mixer.
Gains and step sizes are constants near the top of the script.

## What is on screen

**Frames** (arrows)

| Frame | X | Y | Z |
|-------|---|---|---|
| Body (attached to robot) | red | green | blue |
| World (fixed, drawn at −0.25, −0.25, 0) | orange | cyan | purple |

The world frame is drawn slightly off the origin so it does not overlap the
robot's own frame at start-up. Its axes are still parallel to the true world
axes.

**Overlay (top-left)**
- Body position in the world frame (x, y, z)
- Rotation matrix **R_WB** (3×3), read from MuJoCo's `data.xmat`
- Roll / pitch / yaw (ZYX) recovered from R_WB, as a sanity check
- Controls and colour legend

## Understanding R_WB

R_WB maps vectors from the body frame to the world frame:
`v_world = R_WB · v_body`.

Its **columns** are the body axes expressed in world coordinates, which is
exactly what the arrows show:

- column 1 → red arrow (X_B)
- column 2 → green arrow (Y_B)
- column 3 → blue arrow (Z_B)

Things to try:

- **Start state, TurtleBot:** the robot starts at yaw 90°
  (`INITIAL_YAW_DEG`), so R_WB ≈ `[[0,-1,0],[1,0,0],[0,0,1]]` and the red
  arrow points along world +Y.
- **Drive the TurtleBot (A/D):** only yaw changes. The third row/column stays
  `[0,0,1]` and the blue arrow stays vertical.
- **Fly the quadrotor (I/D):** the matrix gains roll/pitch terms and the blue
  arrow tilts away from vertical. Pressing X brings it back to identity-like.
- **Check:** every column has length 1, columns are perpendicular, and
  det(R) = 1.

## Configuration

| Setting | File | Meaning |
|---------|------|---------|
| `INITIAL_ROLL/PITCH/YAW_DEG` | both | Starting attitude of the robot (the drawn frame follows it) |
| `FORWARD_SPEED`, `TURN_SPEED` | TurtleBot | Wheel speed in rad/s (limited to the actuator's ±7.88) |
| `*_STEP`, `MAX_*`, `TARGET_DECAY` | Quadrotor | Key sensitivity, limits, and how fast targets relax |
| `KP_*`, `KD_ATT` | Quadrotor | Controller gains |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: Model file not found` | Keep the folder layout above; do not move the scripts out of their folders |
| `RuntimeError: 'xyz' not found` | The model XML was changed – restore the bundled files in `models/` |
| macOS: viewer crashes or "launch_passive requires mjpython" | Run with `mjpython` |
| Keys do nothing | Click the viewer window to give it focus |
| Simulation looks slow | Close other heavy apps; the loop paces itself to real time |

## Credits & licences

Models come from the MuJoCo Menagerie ecosystem and keep their original
licences (Apache-2.0), included inside each `models/*` folder:

- Skydio X2 – Google DeepMind MuJoCo Menagerie
- TurtleBot3 Waffle Pi – ROBOTIS MuJoCo Menagerie
