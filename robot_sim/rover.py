"""Reproducible Gazebo adapter for the supplied 500 g educational rover.

No geometry scaling, estimated suspension, or measured-performance claims.
Original input folders are never rewritten.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "robot_sim/models/yandex_rover"
MODEL_NAME = "yandex_rover"
# Simulation assumptions; these are not manufacturer measurements.
SETTINGS = {
    "linear_max": 0.20, "reverse_max": 0.12, "angular_max": 1.0,
    "linear_acceleration": 0.30, "linear_deceleration": 0.50,
    "angular_acceleration": 1.5, "command_timeout_wall_s": 0.6,
    "wheel_velocity_max": 25.0, "drive_torque_max_Nm": 0.03,
    "velocity_p_gain": 0.02, "velocity_i_gain": 0.03, "drive_friction": 0.9,
    "passive_friction": 0.01, "joint_damping": 0.000002,
    "step_size_s": 0.001,
}
SENSOR_POSES = {"laser": [0.035, 0, 0.14],
                "camera_link": [0.082, 0, 0.082], "imu_link": [0, 0, 0.04]}


def add(parent, tag, value=None, **attrs):
    element = ET.SubElement(parent, tag, attrs)
    if value is not None:
        element.text = str(value)
    return element


def numbers(values):
    return " ".join(format(float(value), ".12g") for value in values)


def origin(source):
    return numbers([*map(float, source.get("xyz", "0 0 0").split()),
                    *map(float, source.get("rpy", "0 0 0").split())]) if source is not None else "0 0 0 0 0 0"


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)


def camera(link, name, pose, topic, width=320, height=240, rate=5, optical="camera_optical"):
    sensor = add(link, "sensor", name=name, type="camera")
    add(sensor, "pose", pose)
    add(sensor, "topic", topic)
    add(sensor, "update_rate", rate)
    add(sensor, "always_on", "true")
    add(sensor, "gz_frame_id", optical)
    cam = add(sensor, "camera")
    add(cam, "horizontal_fov", math.radians(100))
    image = add(cam, "image")
    for key, val in {"width": width, "height": height, "format": "R8G8B8"}.items():
        add(image, key, val)
    clip = add(cam, "clip")
    add(clip, "near", 0.005)
    add(clip, "far", 20)
    add(cam, "camera_info_topic", topic.replace("/image", "/camera_info"))


def complete_normals(path):
    lines = path.read_text().splitlines()
    if any(line.startswith("vn ") for line in lines):
        return
    vertices = [list(map(float, line.split()[1:4])) for line in lines if line.startswith("v ")]
    normals = [[0.0, 0.0, 0.0] for _ in vertices]
    for line in lines:
        if not line.startswith("f "):
            continue
        indices = [int(token.split("/")[0])-1 for token in line.split()[1:]]
        a, b, c = [vertices[i] for i in indices[:3]]
        u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        for index in indices:
            for axis in range(3):
                normals[index][axis] += n[axis]
    output = [line for line in lines if not line.startswith("f ")]
    for n in normals:
        length = math.sqrt(sum(x*x for x in n))
        n = [x/length for x in n] if length > 1e-15 else [0,0,1]
        output.append("vn " + numbers(n))
    for line in lines:
        if line.startswith("f "):
            tokens = []
            for token in line.split()[1:]:
                pair = token.split("/")
                tokens.append(pair[0]+"/"+(pair[1] if len(pair)>1 else "")+"/"+pair[0])
            output.append("f " + " ".join(tokens))
    path.write_text("\n".join(output)+"\n")


def build():
    source_dir = ROOT / "rover_urdf"
    source = ET.parse(source_dir / "rover.urdf").getroot()
    config = json.loads((source_dir / "configuration.json").read_text())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir / "meshes", MODEL_DIR / "meshes", dirs_exist_ok=True)
    # Source OBJ has UV coordinates but no vertex normals; Ogre2 cannot build
    # tangent space for textured materials without them.
    for mesh_path in (MODEL_DIR / "meshes").glob("*.obj"):
        complete_normals(mesh_path)
    sdf = ET.Element("sdf", version="1.9")
    model = add(sdf, "model", name=MODEL_NAME, canonical_link="base_link")
    add(model, "self_collide", "false")
    add(model, "static", "false")
    wheel_joints = {joint.find("child").get("link"): joint for joint in source.findall("joint")}
    for src in source.findall("link"):
        name = src.get("name")
        link = add(model, "link", name=name)
        if name in wheel_joints:
            add(link, "pose", origin(wheel_joints[name].find("origin")))
        inertial = add(link, "inertial")
        si = src.find("inertial")
        add(inertial, "pose", origin(si.find("origin")))
        add(inertial, "mass", si.find("mass").get("value"))
        inertia = add(inertial, "inertia")
        for key, value in si.find("inertia").attrib.items():
            add(inertia, key, value)
        for kind in ("visual", "collision"):
            for item in src.findall(kind):
                dst = add(link, kind, name=item.get("name", name + "_" + kind))
                add(dst, "pose", origin(item.find("origin")))
                geometry = copy.deepcopy(item.find("geometry"))
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.attrib.pop("filename")
                    scale = mesh.attrib.pop("scale", "1 1 1")
                    if not (source_dir / filename).is_file():
                        raise FileNotFoundError(filename)
                    add(mesh, "uri", "model://yandex_rover/" + filename)
                    add(mesh, "scale", scale)
                # URDF stores primitive dimensions as attributes; SDF uses children.
                for primitive in geometry:
                    for key, value in list(primitive.attrib.items()):
                        add(primitive, key, value)
                        del primitive.attrib[key]
                dst.append(geometry)
                if kind == "visual":
                    mat = add(dst, "material")
                    if mesh is not None:
                        add(mat, "ambient", "0.7 0.7 0.7 1")
                        add(mat, "diffuse", "1 1 1 1")
                        metal = add(add(mat, "pbr"), "metal")
                        texture = "interior.png" if "interior" in item.get("name", "") else "rover.png"
                        add(metal, "albedo_map", "model://yandex_rover/meshes/" + texture)
                        add(metal, "metalness", 0)
                        add(metal, "roughness", 0.7)
                    else:
                        add(mat, "ambient", "0.88 0.86 0.80 1")
                        add(mat, "diffuse", "0.88 0.86 0.80 1")
                else:
                    middle = "_middle_" in name
                    friction = SETTINGS["drive_friction" if middle else "passive_friction"]
                    if name == "base_link":
                        friction = 0.6
                    ode = add(add(add(dst, "surface"), "friction"), "ode")
                    add(ode, "mu", friction)
                    add(ode, "mu2", friction)
    for src in source.findall("joint"):
        joint = add(model, "joint", name=src.get("name"), type="revolute")
        add(joint, "parent", src.find("parent").get("link"))
        add(joint, "child", src.find("child").get("link"))
        # SDF joint pose defaults to its child link frame, not model frame.
        add(joint, "pose", "0 0 0 0 0 0")
        axis = add(joint, "axis")
        add(axis, "xyz", src.find("axis").get("xyz"))
        limit = add(axis, "limit")
        add(limit, "lower", -1e16)
        add(limit, "upper", 1e16)
        add(limit, "velocity", SETTINGS["wheel_velocity_max"])
        # No zero effort limit on passive joints: they are genuinely unactuated.
        if src.get("name") in config["driven_joints"]:
            add(limit, "effort", SETTINGS["drive_torque_max_Nm"])
        dynamics = add(axis, "dynamics")
        add(dynamics, "damping", SETTINGS["joint_damping"])
        add(dynamics, "friction", 0)
    body = model.find("link[@name='base_link']")
    # Additional mast collision, mass already included in the source body inertial.
    mast = add(body, "collision", name="mast_collision")
    add(mast, "pose", "-0.046 0 0.18 0 0 0")
    cylinder = add(add(mast, "geometry"), "cylinder")
    add(cylinder, "radius", 0.004)
    add(cylinder, "length", 0.13)
    camera(body, "front_camera", numbers(SENSOR_POSES["camera_link"] + [0, 0, 0]),
           "/camera/image")
    camera(body, "overview_camera", "0.36 -0.45 0.30 0 0.30 2.246",
           "/overview/image", 640, 480, 3, "overview_optical")
    body.find("sensor[@name='overview_camera']/camera/horizontal_fov").text = str(math.radians(70))
    lidar = add(body, "sensor", name="lidar", type="gpu_lidar")
    add(lidar, "pose", numbers(SENSOR_POSES["laser"] + [0, 0, 0]))
    for key, value in {"topic": "/scan", "update_rate": 10, "always_on": "true",
                       "gz_frame_id": "laser"}.items():
        add(lidar, key, value)
    scan = add(add(lidar, "lidar"), "scan")
    horizontal = add(scan, "horizontal")
    for key, value in {"samples": 360, "resolution": 1, "min_angle": -math.pi, "max_angle": math.pi}.items():
        add(horizontal, key, value)
    limits = add(lidar.find("lidar"), "range")
    for key, value in {"min": 0.025, "max": 10, "resolution": 0.001}.items():
        add(limits, key, value)
    noise = add(lidar.find("lidar"), "noise")
    for key, value in {"type": "gaussian", "mean": 0, "stddev": 0.002}.items():
        add(noise, key, value)
    imu = add(body, "sensor", name="imu", type="imu")
    add(imu, "pose", numbers(SENSOR_POSES["imu_link"] + [0, 0, 0]))
    for key, value in {"topic": "/imu", "update_rate": 50, "always_on": "true", "gz_frame_id": "imu_link"}.items():
        add(imu, key, value)
    add(imu, "imu")
    contact = add(body, "sensor", name="body_contact", type="contact")
    for key, value in {"topic": "/contacts", "update_rate": 30, "always_on": "true"}.items():
        add(contact, key, value)
    collisions = add(contact, "contact")
    add(collisions, "collision", "body_collision")
    add(collisions, "topic", "/contacts")
    # Gazebo Harmonic Contact reads contact/topic (not sensor/topic).
    # Separate sensors keep body and mast contact channels explicit.
    mast_contact = add(body, "sensor", name="mast_contact", type="contact")
    for key,value in {"topic":"/mast_contacts","update_rate":30,"always_on":"true"}.items():
        add(mast_contact,key,value)
    mast_config = add(mast_contact,"contact")
    add(mast_config,"collision","mast_collision")
    add(mast_config,"topic","/mast_contacts")
    for side in ("left", "right"):
        plugin = add(model, "plugin", filename="gz-sim-joint-controller-system",
                     name="gz::sim::systems::JointController")
        for key, value in {"joint_name": side + "_middle_joint",
                           "topic": "/model/yandex_rover/" + side + "_wheel_velocity",
                           "use_force_commands": "true", "p_gain": SETTINGS["velocity_p_gain"],
                           "i_gain": SETTINGS["velocity_i_gain"], "i_max": 0.003, "i_min": -0.003, "d_gain": 0, "cmd_max": SETTINGS["drive_torque_max_Nm"],
                           "cmd_min": -SETTINGS["drive_torque_max_Nm"]}.items():
            add(plugin, key, value)
    joint_pub = add(model, "plugin", filename="gz-sim-joint-state-publisher-system",
                    name="gz::sim::systems::JointStatePublisher")
    add(joint_pub, "topic", "/model/yandex_rover/joint_state")
    add(joint_pub, "update_rate", 50)
    pose_pub = add(model, "plugin", filename="gz-sim-pose-publisher-system",
                   name="gz::sim::systems::PosePublisher")
    for key, value in {"publish_link_pose": "false", "publish_sensor_pose": "false",
                       "publish_collision_pose": "false", "publish_visual_pose": "false",
                       "publish_model_pose": "true", "publish_nested_model_pose": "false",
                       "use_pose_vector_msg": "false", "update_frequency": 30}.items():
        add(pose_pub, key, value)
    write_xml(sdf, MODEL_DIR / "model.sdf")
    (MODEL_DIR / "model.config").write_text(
        '<?xml version="1.0"?><model><name>Yandex educational mini rover</name><version>1.0</version>'
        '<sdf version="1.9">model.sdf</sdf><description>500 g supplied geometry, estimated physics</description></model>\n')
    # ROS TF model shares exactly the same wheel origins as the SDF.
    tf_model = copy.deepcopy(source)
    for mesh in tf_model.findall(".//mesh"):
        mesh.set("filename", mesh.get("filename"))
    add(tf_model, "link", name="base_footprint")
    fixed = add(tf_model, "joint", name="footprint_to_base", type="fixed")
    add(fixed, "parent", link="base_footprint")
    add(fixed, "child", link="base_link")
    add(fixed, "origin", xyz=numbers([0, 0, config["wheel_radius_m"]]))
    for frame, position in SENSOR_POSES.items():
        add(tf_model, "link", name=frame)
        joint = add(tf_model, "joint", name=frame + "_fixed", type="fixed")
        add(joint, "parent", link="base_link")
        add(joint, "child", link=frame)
        add(joint, "origin", xyz=numbers(position))
    add(tf_model, "link", name="camera_optical")
    joint = add(tf_model, "joint", name="camera_optical_fixed", type="fixed")
    add(joint, "parent", link="camera_link")
    add(joint, "child", link="camera_optical")
    add(joint, "origin", rpy=numbers([-math.pi/2, 0, -math.pi/2]))
    add(tf_model, "link", name="overview_optical")
    joint = add(tf_model, "joint", name="overview_optical_fixed", type="fixed")
    add(joint, "parent", link="base_link")
    add(joint, "child", link="overview_optical")
    # Gazebo camera (+X forward) to ROS optical (+Z forward).
    # Compute composed rotations rather than adding Euler angles.
    add(tf_model, "link", name="overview_camera_link")
    joint.find("parent").set("link", "overview_camera_link")
    add(joint, "origin", rpy=numbers([-math.pi/2,0,-math.pi/2]))
    parent_joint = add(tf_model, "joint", name="overview_camera_fixed", type="fixed")
    add(parent_joint, "parent", link="base_link")
    add(parent_joint, "child", link="overview_camera_link")
    add(parent_joint, "origin", xyz=".36 -.45 .30", rpy="0 .30 2.246")
    write_xml(tf_model, MODEL_DIR / "robot.urdf")
    specification = {**config, "simulation": SETTINGS, "sensor_poses_base_link_m": SENSOR_POSES,
                     "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(source_dir.rglob("*")) if p.is_file() and p.name != ".DS_Store"}}
    (MODEL_DIR / "specification.json").write_text(json.dumps(specification, indent=2))
    for scene in ("warehouse", "apartment", "courtyard"):
        make_world(scene, ROOT / f"robot_sim/worlds/rover_{scene}.sdf")
    import yaml
    (ROOT / "robot_sim/scenes/lab.yaml").write_text(yaml.safe_dump(
        {"name":"lab","spawn":[0,0,0],"goals":[],"obstacles":[]}))
    make_world("lab", ROOT / "robot_sim/worlds/rover_lab.sdf")
    return specification


def make_world(scene, output, spawn=None):
    base_scene = "warehouse" if scene == "lab" else scene
    document = ET.parse(ROOT / f"robot_sim/worlds/{base_scene}.sdf")
    world = document.find("world")
    old_robot = world.find("model[@name='turtlebot3_burger']")
    world.remove(old_robot)
    world.set("name", "rover_lab" if scene == "lab" else "burger_course")
    step = world.find("physics/max_step_size")
    step.text = str(SETTINGS["step_size_s"])
    if world.find("plugin[@name='gz::sim::systems::Imu']") is None:
        add(world, "plugin", filename="gz-sim-imu-system", name="gz::sim::systems::Imu")
    if scene == "lab":
        for item in list(world.findall("model")):
            if item.get("name") != "ground":
                world.remove(item)
        # Static calibration objects are outside the central clear driving area.
        from robot_sim.scripts.build_scenes import obstacle
        for item in [("calibration_red", 2, 0, .25, .4, .25, "red"),
                     ("calibration_blue", 2, 1, .3, .3, .4, "blue"),
                     ("calibration_green", 2, -1, .3, .3, .3, "green"),
                     ("low_step_5mm", 1.5, -1.5, .3, .6, .005, "wood")]:
            world.append(ET.fromstring(obstacle(item)))
    model = ET.parse(MODEL_DIR / "model.sdf").find("model")
    if spawn is None:
        spawn = [0, 0, 0] if scene == "lab" else [-4, -2, 0]
    add(model, "pose", numbers([spawn[0], spawn[1],
                              json.loads((MODEL_DIR/"specification.json").read_text())["wheel_radius_m"] + .0005,
                              0, 0, spawn[2]]))
    world.append(model)
    write_xml(document.getroot(), output)
    return output



def robot_description():
    """Resolve visual resources at launch time, leaving generated URDF portable."""
    document = ET.parse(MODEL_DIR / "robot.urdf")
    for mesh in document.findall(".//mesh"):
        filename = mesh.get("filename")
        if not Path(filename).is_absolute():
            mesh.set("filename", str((MODEL_DIR / filename).resolve()))
    return ET.tostring(document.getroot(), encoding="unicode")


def bridge_configuration():
    import yaml
    bridges = yaml.safe_load((ROOT / "robot_sim/config/navigation_bridge.yaml").read_text())
    bridges = [item for item in bridges if item["ros_topic_name"] not in {"/cmd_vel", "/odom", "/tf"}]
    for item in bridges:
        item["gz_topic_name"] = item["gz_topic_name"].replace("turtlebot3_burger", MODEL_NAME)
    for ros, gz, rt, gt, direction in [
        ("/joint_states", "/model/yandex_rover/joint_state", "sensor_msgs/msg/JointState", "gz.msgs.Model", "GZ_TO_ROS"),
        ("/imu", "/imu", "sensor_msgs/msg/Imu", "gz.msgs.IMU", "GZ_TO_ROS"),
        ("/overview/image_raw", "/overview/image", "sensor_msgs/msg/Image", "gz.msgs.Image", "GZ_TO_ROS"),
    ] + [(f"/rover/{side}_wheel_velocity", f"/model/yandex_rover/{side}_wheel_velocity",
          "std_msgs/msg/Float64", "gz.msgs.Double", "ROS_TO_GZ") for side in ("left", "right")]:
        bridges.append(dict(ros_topic_name=ros, gz_topic_name=gz, ros_type_name=rt, gz_type_name=gt, direction=direction))
    bridges.append(dict(ros_topic_name="/contacts", gz_topic_name="/mast_contacts",
                        ros_type_name="ros_gz_interfaces/msg/Contacts",
                        gz_type_name="gz.msgs.Contacts", direction="GZ_TO_ROS"))
    return bridges


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
