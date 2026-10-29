import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]


class SceneTests(unittest.TestCase):
    def test_mass_inertia_and_support(self):
        model = ET.parse(ROOT / "models/mobile_base/model.sdf").getroot().find("model")
        total_mass = 0
        weighted_center = 0
        for link in model.findall("link"):
            inertial = link.find("inertial")
            mass = float(inertial.findtext("mass"))
            total_mass += mass
            link_pose = [float(value) for value in link.findtext("pose", "0 0 0 0 0 0").split()]
            com_pose = [float(value) for value in inertial.findtext("pose", "0 0 0 0 0 0").split()]
            weighted_center += mass * (link_pose[0] + com_pose[0])
            inertia = inertial.find("inertia")
            principal = [float(inertia.findtext(name)) for name in ("ixx", "iyy", "izz")]
            self.assertTrue(all(value > 0 for value in principal))
            self.assertTrue(all(value <= sum(principal) - value for value in principal))
        self.assertAlmostEqual(total_mass, 1.62)
        self.assertTrue(-0.065 < weighted_center / total_mass < -0.01)

    def test_maps_match_collision_layout_and_goals_are_free(self):
        for scene in ("warehouse", "apartment", "courtyard"):
            metadata = yaml.safe_load((ROOT / f"scenes/{scene}.yaml").read_text())
            occupancy = np.asarray(Image.open(ROOT / f"maps/{scene}.pgm"))
            self.assertEqual(occupancy.shape, (160, 240))
            for obstacle in metadata["obstacles"]:
                column = int((obstacle["x"] + 6) / 0.05)
                row = 159 - int((obstacle["y"] + 4) / 0.05)
                self.assertEqual(occupancy[row, column], 0)
            for goal in [metadata["spawn"], *metadata["goals"]]:
                for delta_x in range(-3, 4):
                    for delta_y in range(-3, 4):
                        column = int((goal[0] + 6) / 0.05) + delta_x
                        row = 159 - int((goal[1] + 4) / 0.05) + delta_y
                        self.assertEqual(occupancy[row, column], 254)

    def test_all_pbr_maps_exist_and_have_detail(self):
        for scene in ("warehouse", "apartment", "courtyard"):
            world = ET.parse(ROOT / f"worlds/{scene}.sdf")
            for tag in ("albedo_map", "normal_map", "roughness_map"):
                maps = world.findall(".//" + tag)
                self.assertGreater(len(maps), 8)
                for element in maps:
                    path = (ROOT / "assets/scene_materials" / element.text.removeprefix("model://scene_materials/")) if element.text.startswith("model://scene_materials/") else Path(element.text)
                    texture = np.asarray(Image.open(path))
                    self.assertEqual(texture.shape[:2], (512, 512))
                    self.assertGreater(float(texture.std()), 0.5)

    def test_controller_limits_match_physics(self):
        parameters = yaml.safe_load((ROOT / "config/nav2.yaml").read_text())
        controller = parameters["controller_server"]["ros__parameters"]["FollowPath"]
        self.assertLessEqual(controller["max_vel_x"], 0.22)
        self.assertLessEqual(controller["max_vel_theta"], 1.2)
        self.assertLessEqual(controller["acc_lim_x"], 0.35)
        self.assertEqual(controller["min_vel_x"], 0.0)


if __name__ == "__main__":
    unittest.main()
