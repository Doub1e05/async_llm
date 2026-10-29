"""Generate reproducible PBR materials, matched collision worlds and occupancy maps."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPAWN = (-4.0, -2.0, 0.0)
SCENES = {
    "warehouse": [
        ("crate_a", -2.8, -2.0, 0.65, 0.9, 0.8, "wood"),
        ("crate_b", -0.3, 0.2, 1.0, 0.7, 0.9, "wood"),
        ("rack", 2.3, 1.5, 0.7, 2.3, 1.2, "metal"),
        ("pallet", -2.0, 1.9, 1.4, 0.8, 0.3, "wood")],
    "apartment": [
        ("sofa", -2.7, -1.9, 0.65, 1.1, 0.7, "fabric"),
        ("partition", 0.0, 1.8, 0.16, 2.0, 1.6, "plaster"),
        ("table", 1.9, -1.4, 1.3, 0.8, 0.65, "wood"),
        ("cabinet", -3.4, 1.8, 0.7, 1.5, 1.4, "wood")],
    "courtyard": [
        ("planter", -2.8, -2.0, 0.6, 0.75, 0.6, "brick"),
        ("bollard_a", -0.2, -0.4, 0.25, 0.25, 0.8, "metal"),
        ("bollard_b", -0.2, 0.6, 0.25, 0.25, 0.8, "metal"),
        ("bench", 2.0, 1.8, 1.8, 0.5, 0.5, "wood"),
        ("island", 2.5, -1.8, 1.1, 0.9, 0.4, "brick")]
}


def materials():
    folder = ROOT / "assets/textures"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(73)
    for name, color in {"wood": (142, 99, 57), "brick": (156, 77, 54),
                        "tiles": (181, 184, 177), "concrete": (125, 132, 133),
                        "asphalt": (67, 73, 77), "metal": (92, 113, 124),
                        "plaster": (208, 197, 177), "fabric": (74, 110, 132),
                        "red": (195, 35, 30), "blue": (30, 75, 200), "green": (35, 155, 55)}.items():
        noise = rng.normal(0, 5, (512, 512))
        if name == "wood":
            noise += 12 * np.sin(np.arange(512)[None, :] * 0.2 + rng.normal(0, 0.1, (512, 1)))
        elif name == "fabric":
            noise += 7 * ((np.indices((512, 512)).sum(axis=0) % 4) < 2)
        albedo = np.clip(np.array(color)[None, None, :] + noise[:, :, None], 0, 255).astype("uint8")
        image = Image.fromarray(albedo)
        draw = ImageDraw.Draw(image)
        if name in {"tiles", "brick"}:
            step = 128 if name == "tiles" else 64
            for row in range(0, 512, step):
                draw.line((0, row, 512, row), fill=(85, 85, 80), width=4)
                offset = (64 if (row // step) % 2 else 0) if name == "brick" else 0
                for column in range(offset, 512, 128):
                    draw.line((column, row, column, row + step), fill=(85, 85, 80), width=4)
        if name in {"red", "blue", "green"}:
            draw.text((35, 190), name.upper(), fill=(250, 250, 245), font_size=85)
        image.save(folder / f"{name}_albedo.png")
        height = np.asarray(image.convert("L"), dtype=float) / 255
        grad_y, grad_x = np.gradient(height)
        normals = np.dstack((-grad_x * 2, -grad_y * 2, np.ones_like(height)))
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        Image.fromarray(((normals + 1) * 127.5).astype("uint8")).save(folder / f"{name}_normal.png")
        roughness = np.clip(205 + noise, 0, 255).astype("uint8")
        Image.fromarray(roughness).save(folder / f"{name}_roughness.png")


def material(name):
    folder = "model://scene_materials/textures"
    return f"""<material><diffuse>1 1 1 1</diffuse><pbr><metal>
        <albedo_map>{folder}/{name}_albedo.png</albedo_map>
        <normal_map>{folder}/{name}_normal.png</normal_map>
        <roughness_map>{folder}/{name}_roughness.png</roughness_map>
        <metalness>{0.6 if name == "metal" else 0.0}</metalness>
        <roughness>0.8</roughness></metal></pbr></material>"""


def obstacle(item):
    name, position_x, position_y, size_x, size_y, size_z, surface = item
    return f"""<model name="{name}"><static>true</static><pose>{position_x} {position_y} {size_z/2} 0 0 0</pose>
    <link name="body"><collision name="collision"><geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry></collision>
    <visual name="visual"><geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry>{material(surface)}</visual></link></model>"""


def marker(color, goal):
    center_x, center_y = goal[0] + 0.5, goal[1] + 0.5
    return f"""<model name="{color}_marker"><static>true</static><pose>{center_x} {center_y} 0.95 0 0 0</pose>
    <link name="board"><collision name="board_collision"><geometry><box><size>0.04 0.55 0.55</size></box></geometry></collision>
    <visual name="board"><geometry><box><size>0.04 0.55 0.55</size></box></geometry>{material(color)}</visual></link></model>"""


def build():
    materials()
    (ROOT / "maps").mkdir(exist_ok=True)
    (ROOT / "scenes").mkdir(exist_ok=True)
    robot = (ROOT / "models/mobile_base/model.sdf").read_text()
    model = ET.fromstring(robot).find("model")
    model_xml = ET.tostring(model, encoding="unicode")
    walls = [("north", 0, 3.5, 12, 0.15, 1.8, "brick"),
             ("south", 0, -3.5, 12, 0.15, 1.8, "brick"),
             ("east", 5.9, 0, 0.15, 7, 1.8, "brick"),
             ("west", -5.9, 0, 0.15, 7, 1.8, "brick")]
    for name, scene_items in SCENES.items():
        goals = [[-1.7, -2.0, 0.0], [1.0, 0.0, 0.0], [4.0, 2.3, 0.0]]
        items = list(scene_items) + [(color+"_stand", goal[0]+0.5, goal[1]+0.5, 0.06, 0.06, 0.9, "metal")
                                    for color,goal in zip(("red","blue","green"), goals)]
        floor = {"warehouse": "concrete", "apartment": "tiles", "courtyard": "asphalt"}[name]
        ground = obstacle(("ground", 0, 0, 12, 8, 0.05, floor)).replace("<pose>0 0 0.025", "<pose>0 0 -0.025")
        objects = "".join(obstacle(item) for item in walls + items)
        markers = "".join(marker(color, goal) for color,goal in zip(("red","blue","green"), goals))
        world = f"""<?xml version="1.0"?><sdf version="1.9"><world name="burger_course">
        <physics name="physics" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor></physics>
        <gravity>0 0 -9.81</gravity>
        <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
        <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
        <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
        <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
        <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.65 0.76 0.87 1</background><shadows>true</shadows></scene>
        <light name="sun" type="directional"><pose>0 0 6 0 0 0</pose><direction>-0.4 0.2 -1</direction><diffuse>0.9 0.9 0.85 1</diffuse><cast_shadows>true</cast_shadows></light>
        {ground}{objects}{markers}{model_xml}</world></sdf>"""
        (ROOT / "worlds" / f"{name}.sdf").write_text(world)
        resolution = 0.05
        width, height = 240, 160
        occupancy = np.full((height, width), 254, dtype=np.uint8)
        for _, center_x, center_y, size_x, size_y, _, _ in walls + items:
            minimum_x = max(0, int(math.floor((center_x - size_x/2 + 6) / resolution)))
            maximum_x = min(width, int(math.ceil((center_x + size_x/2 + 6) / resolution)))
            minimum_y = max(0, int(math.floor((center_y - size_y/2 + 4) / resolution)))
            maximum_y = min(height, int(math.ceil((center_y + size_y/2 + 4) / resolution)))
            occupancy[height-maximum_y:height-minimum_y, minimum_x:maximum_x] = 0
        Image.fromarray(occupancy).save(ROOT / "maps" / f"{name}.pgm")
        metadata = {"image": f"{name}.pgm", "mode": "trinary", "resolution": resolution,
                    "origin": [-6.0, -4.0, 0.0], "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.25}
        (ROOT / "maps" / f"{name}.yaml").write_text(yaml.safe_dump(metadata))
        description = {"name": name, "spawn": list(SPAWN),
                       "goals": goals,
                       "obstacles": [dict(zip(("name","x","y","width","depth","height","material"), item)) for item in items]}
        (ROOT / "scenes" / f"{name}.yaml").write_text(yaml.safe_dump(description))
        print(f"Built {name}: {len(items)} obstacles, map {width}x{height}")


if __name__ == "__main__":
    build()
