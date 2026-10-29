"""macOS: .venv/bin/mjpython run.py --linear 0.1 --angular 0"""
from pathlib import Path
import argparse,json,time
import mujoco
import mujoco.viewer
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--linear',type=float,default=0.0,help='m/s')
parser.add_argument('--angular',type=float,default=0.0,help='rad/s')
args=parser.parse_args()
p=json.loads((ROOT/'parameters.json').read_text())
m=mujoco.MjModel.from_xml_path(str(ROOT/'delivery_robot.xml'));d=mujoco.MjData(m)
d.ctrl[:]=[(args.linear-args.angular*p['track_m']/2)/p['wheel_radius_m'],(args.linear+args.angular*p['track_m']/2)/p['wheel_radius_m']]
with mujoco.viewer.launch_passive(m,d) as viewer:
    viewer.cam.distance=.6;viewer.cam.lookat[:]=[0,0,.08];viewer.cam.azimuth=135;viewer.cam.elevation=-20
    while viewer.is_running():
        start=time.monotonic();mujoco.mj_step(m,d);viewer.sync()
        time.sleep(max(0,m.opt.timestep-(time.monotonic()-start)))
