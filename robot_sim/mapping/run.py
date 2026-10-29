"""Own a complete sensor-only SLAM experiment; truth goes to evaluation only."""
import argparse
import hashlib
import shutil
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
import yaml

from robot_sim.rover import ROOT,MODEL_DIR,make_world,bridge_configuration,add,write_xml,robot_description


def prepare(output):
    world=make_world("warehouse",output/"world.sdf",spawn=[-4,-2.8,0])
    tree=ET.parse(world);parent=tree.find("world")
    actor=add(parent,"model",name="moving_box")
    add(actor,"pose","-1.8 -2.8 .225 0 0 0")
    link=add(actor,"link",name="body")
    add(link,"kinematic","true");add(link,"gravity","false")
    inertial=add(link,"inertial");add(inertial,"mass",5)
    inertia=add(inertial,"inertia")
    for key,val in {"ixx":.11,"iyy":.11,"izz":.05,"ixy":0,"ixz":0,"iyz":0}.items():add(inertia,key,val)
    for kind in ("visual","collision"):
        elem=add(link,kind,name=kind)
        add(add(add(elem,"geometry"),"box"),"size",".25 .25 .45")
        if kind=="visual":
            mat=add(elem,"material")
            add(mat,"ambient",".8 .65 .03 1");add(mat,"diffuse","1 .8 .04 1")
    # Less ideal sensor rates, all retained original model physics/geometry.
    model=parent.find("model[@name='yandex_rover']")
    model.find("link[@name='base_link']/sensor[@name='overview_camera']/update_rate").text="1"
    write_xml(tree.getroot(),world)
    bridges=bridge_configuration()
    rename={"/scan":"/sim/scan","/imu":"/sim/imu","/camera/image_raw":"/sim/camera/image",
            "/camera/camera_info":"/sim/camera/camera_info","/joint_states":"/sim/joints",
            "/ground_truth/pose":"/evaluation/ground_truth"}
    for item in bridges:
        item["ros_topic_name"]=rename.get(item["ros_topic_name"],item["ros_topic_name"])
    (output/"bridge.yaml").write_text(yaml.safe_dump(bridges))
    prefix=Path(os.environ["CONDA_PREFIX"])
    config=yaml.safe_load((prefix/"share/slam_toolbox/config/mapper_params_online_async.yaml").read_text())
    config["slam_toolbox"]["ros__parameters"].update(
        use_sim_time=True,scan_topic="/scan/slam",resolution=.04,max_laser_range=6.0,
        min_laser_range=.12,minimum_time_interval=.2,minimum_travel_distance=.04,
        minimum_travel_heading=.05,scan_queue_size=1,map_update_interval=1.0,
        transform_publish_period=.05,transform_timeout=.2,enable_interactive_mode=False,
        scan_buffer_size=20,loop_match_minimum_chain_size=8)
    (output/"slam.yaml").write_text(yaml.safe_dump(config))
    (output/"robot_state_publisher.yaml").write_text(yaml.safe_dump({"robot_state_publisher":{"ros__parameters":{
        "use_sim_time":True,"publish_frequency":30.0,"robot_description":robot_description()}}}))
    (output/"evaluation_scenario.json").write_text(json.dumps({
        "world":"burger_course","spawn":[-4,-2.8,0],"moving_box_initial":[-1.8,-2.8],
        "moving_box_final":[-1.8,-.65],"actor_start_sim_s":5,"actor_speed_m_s":.25},indent=2))
    return world


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--duration",type=float,default=150)
    parser.add_argument("--seed",type=int,default=29)
    parser.add_argument("--record-bag",action="store_true")
    args=parser.parse_args()
    output=(args.output_dir or ROOT/"robot_sim/results"/("mapping_"+datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if (output/"mapping.jsonl").exists():parser.error("Existing recording; choose a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    prefix=Path(os.environ["CONDA_PREFIX"])
    world=prepare(output)
    shutil.copy2(ROOT/"robot_sim/config/mapping_sensors.yaml",output/"sensor_profile.yaml")
    shutil.copy2(ROOT/"robot_sim/config/mapping_ekf.yaml",output/"ekf.yaml")
    source_files=list((ROOT/"robot_sim/mapping").glob("*.py"))+[ROOT/"robot_sim/scripts/rover_drive.py"]
    (output/"source_hashes.json").write_text(json.dumps({
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},indent=2))
    processes=[];logs=[]
    def start(name,cmd):
        log=(output/(name+".log")).open("w");logs.append(log)
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,process));return process
    def interrupted(*_):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    print("Mapping output:",output,flush=True)
    code=1
    try:
        start("gazebo",["xvfb-run","-a","-s","-screen 0 640x480x24 -nolisten tcp","gz","sim","-s","-r",str(world)])
        start("bridge",[str(prefix/"lib/ros_gz_bridge/parameter_bridge"),"--ros-args","-p","config_file:="+str(output/"bridge.yaml")])
        start("drive",[sys.executable,"robot_sim/scripts/rover_drive.py","--output",str(output),
                       "--odom-topic","/wheel/ideal","--no-odom-tf","--ros-args","-r","/joint_states:=/sim/joints"])
        start("tf",[str(prefix/"lib/robot_state_publisher/robot_state_publisher"),"--ros-args","--params-file",str(output/"robot_state_publisher.yaml"),
                    "-r","/joint_states:=/sim/joints"])
        start("sensors",[sys.executable,"-m","robot_sim.mapping.sensors","--output",str(output),"--seed",str(args.seed)])
        start("ekf",[str(prefix/"lib/robot_localization/ekf_node"),"--ros-args","--params-file",str(ROOT/"robot_sim/config/mapping_ekf.yaml"),
                     "-r","odometry/filtered:=/odom"])
        start("slam",["ros2","launch","slam_toolbox","online_async_launch.py",
                      "slam_params_file:="+str(output/"slam.yaml"),"use_sim_time:=true"])
        start("mapper",[sys.executable,"-m","robot_sim.mapping.mapper","--output",str(output)])
        if args.record_bag:
            start("bag",["ros2","bag","record","--use-sim-time","--storage","sqlite3","-o",str(output/"sensors_bag"),
                         "/scan","/imu/data","/wheel/odometry","/odom","/camera/image_raw",
                         "/camera/camera_info","/tf","/tf_static"])
        client=start("experiment",[sys.executable,"-m","robot_sim.mapping.experiment",
                                    "--output",str(output),"--duration",str(args.duration)])
        deadline=time.monotonic()+max(360,args.duration*12)
        while client.poll() is None:
            for name,process in processes:
                if process is not client and process.poll() is not None:
                    raise RuntimeError(name+" exited; see "+str(output/(name+".log")))
            if time.monotonic()>deadline:raise TimeoutError("Mapping experiment wall deadline")
            time.sleep(.3)
        code=client.returncode
    except (KeyboardInterrupt,Exception) as e:
        (output/"runner_error.txt").write_text(str(e))
        print("Stopped:",e,flush=True)
    finally:
        for _,process in reversed(processes):
            try:os.killpg(process.pid,signal.SIGINT)
            except ProcessLookupError:pass
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and any(p.poll() is None for _,p in processes):time.sleep(.1)
        for _,process in processes:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            process.wait()
        for f in logs:f.close()
        (output/"processes.json").write_text(json.dumps(
            [{"name":n,"pid":p.pid,"returncode":p.returncode} for n,p in processes],indent=2))
    if (output/"summary.json").exists():print((output/"summary.json").read_text(),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
