"""Build a physically consistent surrogate; values are assumptions, not calibration."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def text(parent, path, value):
    element = parent.find(path)
    if element is None:
        element = ET.SubElement(parent, path)
    element.text = str(value)
    return element


def configure():
    model = ET.parse(ROOT / "worlds/burger_course.sdf").getroot().find("world/model[@name='turtlebot3_burger']")
    text(model, "pose", "-4 -2 0.078 0 0 0")
    body = model.find("link[@name='base_link']")
    text(body.find("inertial"), "pose", "-0.025 0 -0.015 0 0 0")
    mass, width, depth, height = 1.5, 0.178, 0.135, 0.09
    for name, value in {"ixx": mass*(depth**2+height**2)/12,
                        "iyy": mass*(width**2+height**2)/12,
                        "izz": mass*(width**2+depth**2)/12}.items():
        text(body.find("inertial/inertia"), name, value)
    lidar = body.find("sensor[@name='lidar']")
    text(lidar, "gz_frame_id", "laser")
    noise = ET.SubElement(lidar.find("lidar"), "noise")
    text(noise, "type", "gaussian")
    text(noise, "mean", 0)
    text(noise, "stddev", 0.003)
    camera = ET.fromstring("""
      <sensor name="front_camera" type="camera">
        <pose>0.09 0 0.10 0 0 0</pose><topic>/camera/image</topic>
        <update_rate>5</update_rate><always_on>true</always_on>
        <gz_frame_id>camera_optical</gz_frame_id>
        <camera><horizontal_fov>1.74532925</horizontal_fov>
          <image><width>320</width><height>240</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>20</far></clip>
          <camera_info_topic>/camera/camera_info</camera_info_topic>
        </camera>
      </sensor>""")
    body.append(camera)
    body.append(ET.fromstring("""
      <sensor name="body_contact" type="contact"><always_on>true</always_on>
        <update_rate>20</update_rate><topic>/contacts</topic>
        <contact><collision>body_collision</collision></contact>
      </sensor>"""))
    for side, lateral in [("left", 0.095), ("right", -0.095)]:
        wheel = model.find(f"link[@name='{side}_wheel']")
        text(wheel, "pose", f"0 {lateral} -0.045 0 0 0")
        for element in [wheel.find("collision"), wheel.find("visual")]:
            text(element, "pose", f"0 0 0 {math.pi/2} 0 0")
        wheel_mass, radius, length = 0.05, 0.033, 0.018
        inertia = wheel.find("inertial/inertia")
        transverse = wheel_mass * (3*radius**2 + length**2)/12
        for name, value in {"ixx": transverse, "iyy": wheel_mass*radius**2/2,
                            "izz": transverse}.items():
            text(inertia, name, value)
        surface = wheel.find("collision/surface")
        text(surface.find("friction/ode"), "mu", 0.9)
        text(surface.find("friction/ode"), "mu2", 0.9)
        text(surface.find("friction/ode"), "slip1", 0.0001)
        text(surface.find("friction/ode"), "slip2", 0.0001)
        joint = model.find(f"joint[@name='{side}_wheel_joint']")
        text(joint, "pose", "0 0 0 0 0 0")
        text(joint.find("axis/limit"), "effort", 0.4)
        text(joint.find("axis/limit"), "velocity", 8.0)
        dynamics = ET.SubElement(joint.find("axis"), "dynamics")
        text(dynamics, "damping", 0.002)
        text(dynamics, "friction", 0.0)
    caster = model.find("link[@name='caster']")
    text(caster, "pose", "-0.065 0 -0.066 0 0 0")
    inertia = caster.find("inertial/inertia")
    for name in ("ixx", "iyy", "izz"):
        text(inertia, name, 2 * 0.02 * 0.012**2 / 5)
    caster.find("collision").append(ET.fromstring(
        "<surface><friction><ode><mu>0.01</mu><mu2>0.01</mu2></ode></friction></surface>"))
    drive = model.find("plugin[@name='gz::sim::systems::DiffDrive']")
    text(drive, "child_frame_id", "base_footprint")
    for name, value in {"max_linear_velocity": 0.22, "min_linear_velocity": -0.12,
                        "max_angular_velocity": 1.2, "min_angular_velocity": -1.2,
                        "max_linear_acceleration": 0.35, "min_linear_acceleration": -0.5,
                        "max_angular_acceleration": 1.5, "min_angular_acceleration": -1.5,
                        "odom_publish_frequency": 30}.items():
        text(drive, name, value)
    model.append(ET.fromstring("""
      <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose><publish_sensor_pose>false</publish_sensor_pose>
        <publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose>
        <publish_nested_model_pose>false</publish_nested_model_pose><publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>false</use_pose_vector_msg><update_frequency>20</update_frequency>
      </plugin>"""))
    sdf = ET.Element("sdf", {"version": "1.9"})
    sdf.append(model)
    folder = ROOT / "models/mobile_base"
    folder.mkdir(parents=True, exist_ok=True)
    ET.indent(sdf)
    ET.ElementTree(sdf).write(folder / "model.sdf", encoding="unicode", xml_declaration=True)
    (folder / "model.config").write_text(
        '<?xml version="1.0"?><model><name>Mobile base surrogate</name><version>1.0</version>'
        '<sdf version="1.9">model.sdf</sdf><description>Parameterized differential drive test platform</description></model>\n')
    print("Robot generated: mass 1.62 kg; wheel radius .033 m; track .19 m; body z .078 m")


if __name__ == "__main__":
    configure()
