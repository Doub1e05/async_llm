"""Generate a simulator-neutral URDF from the prepared Yandex rover scene."""
from pathlib import Path
import json, shutil, math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'yandex_rover/mujoco'
scene = ET.parse(SOURCE / 'delivery_robot.xml').getroot()
source_base = scene.find('worldbody/body')
meshdir = ROOT / 'meshes'
meshdir.mkdir(exist_ok=True)
for p in (SOURCE / 'meshes').glob('*.png'):
    shutil.copy2(p, meshdir / p.name)
for p in (SOURCE / 'meshes').glob('*.obj'):
    material = 'interior_surface' if p.stem == 'interior' else 'rover_surface'
    (meshdir / p.name).write_text(f'mtllib rover.mtl\nusemtl {material}\n' + p.read_text())
(meshdir / 'rover.mtl').write_text('''newmtl rover_surface
Ka 0.2 0.2 0.2
Kd 1 1 1
Ks 0.1 0.1 0.1
Ns 20
d 1
map_Kd rover.png

newmtl interior_surface
Ka 0.2 0.2 0.2
Kd 1 1 1
Ks 0.1 0.1 0.1
Ns 20
d 1
map_Kd interior.png
''')

def sub(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kw.items()})

def inertial(link, source):
    src = source.find('inertial')
    node = sub(link, 'inertial')
    sub(node, 'origin', xyz=src.get('pos', '0 0 0'), rpy='0 0 0')
    sub(node, 'mass', value=src.get('mass'))
    ixx, iyy, izz = src.get('diaginertia').split()
    sub(node, 'inertia', ixx=ixx, iyy=iyy, izz=izz, ixy='0', ixz='0', iyz='0')

def visual(link, name):
    node = sub(link, 'visual', name=f'{name}_visual')
    sub(node, 'origin', xyz='0 0 0', rpy='0 0 0')
    source_mesh = scene.find(f"asset/mesh[@name='{name}']")
    sub(sub(node, 'geometry'), 'mesh', filename=f'meshes/{name}.obj', scale=source_mesh.get('scale', '1 1 1'))
    # Textures are described by OBJ/MTL, with a neutral URDF fallback.
    sub(node, 'material', name='neutral_fallback')

robot = ET.Element('robot', name='yandex_rover')
robot.append(ET.Comment('SI units. Six fixed wheel axes; only middle joints should be actuated. No simulator-specific plugins.'))
sub(sub(robot, 'material', name='neutral_fallback'), 'color', rgba='0.75 0.75 0.75 1')
base = sub(robot, 'link', name='base_link')
inertial(base, source_base)
visual(base, 'body'); visual(base, 'interior')
# Standard URDF has no capsule or emissive material. Use a cylinder with
# spherical ends, matching the MuJoCo inserts, and the same diffuse color.
lamp_material = scene.find("asset/material[@name='headlight_white']")
if lamp_material is not None:
    sub(sub(robot, 'material', name='headlight_milk'), 'color', rgba=lamp_material.get('rgba'))
    for lamp in source_base.findall("geom[@material='headlight_white']"):
        coords = list(map(float, lamp.get('fromto').split()))
        start, end = coords[:3], coords[3:]
        delta = [b-a for a, b in zip(start, end)]
        length = math.sqrt(sum(x*x for x in delta))
        pitch = math.atan2(math.hypot(delta[0], delta[1]), delta[2])
        yaw = math.atan2(delta[1], delta[0])
        center = [(a+b)/2 for a, b in zip(start, end)]
        radius = lamp.get('size')
        for suffix, position, kind in [('center', center, 'cylinder'), ('end0', start, 'sphere'), ('end1', end, 'sphere')]:
            vis = sub(base, 'visual', name=f'{lamp.get("name")}_{suffix}')
            sub(vis, 'origin', xyz=' '.join(map(str, position)), rpy=f'0 {pitch} {yaw}' if kind == 'cylinder' else '0 0 0')
            geom = sub(vis, 'geometry')
            if kind == 'cylinder':
                sub(geom, 'cylinder', radius=radius, length=length)
            else:
                sub(geom, 'sphere', radius=radius)
            sub(vis, 'material', name='headlight_milk')
src_collision = source_base.find("geom[@name='body_collision']")
c = sub(base, 'collision', name='body_collision')
sub(c, 'origin', xyz=src_collision.get('pos'), rpy='0 0 0')
sub(sub(c, 'geometry'), 'box', size=' '.join(str(2 * float(x)) for x in src_collision.get('size').split()))
wheel_info = {}
for body in source_base.findall('body'):
    name = body.get('name').removesuffix('_wheel')
    link = sub(robot, 'link', name=f'{name}_wheel')
    inertial(link, body); visual(link, name)
    radius, halfwidth = body.find("geom[@type='cylinder']").get('size').split()
    c = sub(link, 'collision', name=f'{name}_collision')
    sub(c, 'origin', xyz='0 0 0', rpy=f'{math.pi/2} 0 0')
    sub(sub(c, 'geometry'), 'cylinder', radius=radius, length=2 * float(halfwidth))
    j = sub(robot, 'joint', name=f'{name}_joint', type='continuous')
    sub(j, 'parent', link='base_link'); sub(j, 'child', link=f'{name}_wheel')
    sub(j, 'origin', xyz=body.get('pos'), rpy='0 0 0')
    sub(j, 'axis', xyz='0 1 0')
    drive = 'middle' in name
    sub(j, 'limit', effort='0.03' if drive else '0', velocity='25')
    sub(j, 'dynamics', damping='0.000002', friction='0')
    wheel_info[name] = {'position_m': list(map(float, body.get('pos').split())), 'driven': drive,
                        'radius_m': float(radius),
                        'sliding_friction': float(body.find("geom[@type='cylinder']").get('friction').split()[0])}
ET.indent(robot, space='  ')
ET.ElementTree(robot).write(ROOT / 'rover.urdf', encoding='utf-8', xml_declaration=True)
settings = {
    'wheel_radius_m': wheel_info['left_middle']['radius_m'],
    'passive_wheel_radius_m': wheel_info['left_front']['radius_m'],
    'wheel_width_m': 2 * float(halfwidth),
    'track_m': wheel_info['left_middle']['position_m'][1] - wheel_info['right_middle']['position_m'][1],
    'total_mass_kg': 0.5,
    'driven_joints': ['left_middle_joint', 'right_middle_joint'],
    'passive_joints': [f'{k}_joint' for k, v in wheel_info.items() if not v['driven']],
    'wheel_layout': wheel_info,
    'notes': 'Mass distribution, inertia, damping and motor limits are estimates. SI units. Free base and motor controllers configured in the simulator.'
}
(ROOT / 'configuration.json').write_text(json.dumps(settings, indent=2) + '\n')
contact_settings = {
    'note': 'Apply in the target simulator: standard URDF does not encode portable ground friction. These are simulation assumptions, not measured tyre properties.',
    'links': {f'{name}_wheel': {'sliding_friction': info['sliding_friction'], 'passive': not info['driven']}
              for name, info in wheel_info.items()},
    'mujoco': {'wheel_contact_priority': 1, 'velocity_gain': 0.02,
               'note': 'priority=1 prevents floor friction from overriding the wheel value. Set only central motor actuators.'}
}
(ROOT / 'contact_settings.json').write_text(json.dumps(contact_settings, indent=2) + '\n')
shutil.copy2(ROOT.parent / 'yandex_rover/provenance.json', ROOT / 'provenance.json')
print(ROOT / 'rover.urdf')
