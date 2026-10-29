"""Checks for scale, physical structure and nontrivial encoder kinematics."""
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from robot_sim.rover import MODEL_DIR,robot_description
from robot_sim.rover_control import integrate_encoders, wheel_speeds

ROOT = Path(__file__).resolve().parents[2]


class RoverTests(unittest.TestCase):
    def test_supplied_dimensions_and_mass_preserved(self):
        original = ET.parse(ROOT/"rover_urdf/rover.urdf")
        model = ET.parse(MODEL_DIR/"model.sdf")
        self.assertEqual(len(model.findall("./model/joint")), 6)
        self.assertAlmostEqual(sum(float(x.text) for x in model.findall(".//inertial/mass")), .5)
        for src in original.findall("./link"):
            dst = model.find("./model/link[@name='"+src.get("name")+"']")
            for key, val in src.find("inertial/inertia").attrib.items():
                self.assertAlmostEqual(float(dst.findtext("inertial/inertia/"+key)),float(val))
            geometry = src.find("collision/geometry")
            if geometry is not None:
                for primitive in geometry:
                    for key, value in primitive.attrib.items():
                        actual=dst.findtext("collision/geometry/"+primitive.tag+"/"+key)
                        self.assertEqual([float(x) for x in actual.split()], [float(x) for x in value.split()])

    def test_description_resolves_meshes_without_hardcoded_user_path(self):
        portable=ET.parse(MODEL_DIR/"robot.urdf")
        self.assertTrue(all(not Path(m.get("filename")).is_absolute() for m in portable.findall(".//mesh")))
        resolved=ET.fromstring(robot_description())
        self.assertTrue(all(Path(m.get("filename")).is_file() for m in resolved.findall(".//mesh")))

    def test_torque_bounded_and_only_two_actuated_wheels(self):
        model = ET.parse(MODEL_DIR/"model.sdf")
        self.assertFalse(model.findall(".//plugin[@name='gz::sim::systems::DiffDrive']"))
        controllers=model.findall(".//plugin[@name='gz::sim::systems::JointController']")
        self.assertEqual({p.findtext("joint_name") for p in controllers},
                         {"left_middle_joint","right_middle_joint"})
        for p in controllers:
            self.assertEqual(p.findtext("use_force_commands"),"true")
            self.assertAlmostEqual(float(p.findtext("cmd_max")),.03)
            self.assertAlmostEqual(float(p.findtext("cmd_min")),-.03)

    def test_contact_channels_and_collision_associations(self):
        model=ET.parse(MODEL_DIR/"model.sdf")
        monitored=set()
        for sensor in model.findall(".//sensor[@type='contact']"):
            collisions=sensor.findall("contact/collision")
            self.assertEqual(len(collisions),1)
            monitored.add(collisions[0].text)
            self.assertEqual(sensor.findtext("contact/topic"),sensor.findtext("topic"))
        self.assertEqual(monitored,{"body_collision","mast_collision"})

    def test_sensor_frames_exist_in_tf_tree(self):
        sdf=ET.parse(MODEL_DIR/"model.sdf")
        urdf=ET.parse(MODEL_DIR/"robot.urdf")
        frames={x.get("name") for x in urdf.findall("./link")}
        for sensor in sdf.findall(".//sensor"):
            frame=sensor.findtext("gz_frame_id")
            if frame and frame != "overview_optical":
                self.assertIn(frame,frames)
        self.assertIn("base_footprint",frames)

    def test_mesh_normals_and_resources(self):
        for p in (MODEL_DIR/"meshes").glob("*.obj"):
            lines=p.read_text().splitlines()
            normals=[list(map(float,line.split()[1:])) for line in lines if line.startswith("vn ")]
            self.assertTrue(normals,p.name)
            self.assertTrue(all(abs(sum(v*v for v in n)-1)<1e-6 for n in normals))
            self.assertTrue(all(len(t.split("/"))==3 for line in lines if line.startswith("f ") for t in line.split()[1:]))
        for p in ET.parse(MODEL_DIR/"model.sdf").findall(".//uri")+ET.parse(MODEL_DIR/"model.sdf").findall(".//albedo_map"):
            self.assertTrue((MODEL_DIR/p.text.removeprefix("model://yandex_rover/")).is_file())

    def test_encoder_exact_arc_and_inverse(self):
        # Radius=1 m and track=2 m, wheel travels 0 and pi:
        # radius-one left arc by pi/2 ends at (1,1,pi/2).
        self.assertEqual(integrate_encoders((0,0,0),0,0,1,2),(0,0,0))
        pose=integrate_encoders((0,0,0),0,math.pi,1,2)
        for actual,expected in zip(pose,(1,1,math.pi/2)):
            self.assertAlmostEqual(actual,expected)
        back=integrate_encoders(pose,0,-math.pi,1,2)
        for val in back:self.assertAlmostEqual(val,0)

    def test_speed_saturation_preserves_curvature(self):
        left,right=wheel_speeds(10,20,.02,.1,25)
        self.assertLessEqual(max(abs(left),abs(right)),25)
        self.assertAlmostEqual(right/left,11/9)
        self.assertEqual(wheel_speeds(float("nan"),0,.02,.1,25),(0,0))
        left,right=wheel_speeds(0,.65,.022124,.11535586,25)
        self.assertLess(left,0)
        self.assertAlmostEqual(left,-right)


if __name__=="__main__":
    unittest.main()
