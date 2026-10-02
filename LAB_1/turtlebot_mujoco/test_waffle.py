import mujoco
import mujoco.viewer

xml_path = "/home/pratham/robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
