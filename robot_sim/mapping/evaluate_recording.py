"""Offline timestamp matching of recorded estimated TF and evaluation-only truth.

No fitted rigid alignment; the fixed initial spawn defines the reference.
This avoids optimistic evaluation from only the few online TF matches.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
import rosbag2_py


def evaluate(folder):
    rclpy.init()
    buffer=Buffer(cache_time=Duration(seconds=300))
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(folder/"sensors_bag"),storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr","cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf","/tf_static"]))
    counts={}
    while reader.has_next():
        topic,raw,_=reader.read_next()
        msg=deserialize_message(raw,TFMessage)
        for tf in msg.transforms:
            key=tf.header.frame_id+"->"+tf.child_frame_id
            counts[key]=counts.get(key,0)+1
            if topic=="/tf_static":buffer.set_transform_static(tf,"recorded")
            else:buffer.set_transform(tf,"recorded")
    samples=[json.loads(line) for line in (folder/"evaluation_trajectory.jsonl").read_text().splitlines()]
    pairs=[];missed=0
    for sample in samples:
        now=sample["sim_time"]
        stamp=Time(seconds=int(now),nanoseconds=int((now%1)*1e9))
        try:
            tf=buffer.lookup_transform("map","base_footprint",stamp)
            p=tf.transform.translation;q=tf.transform.rotation
            estimate=[p.x,p.y,2*math.atan2(q.z,q.w)]
            truth=[sample["truth"][0]+4,sample["truth"][1]+2.8,sample["truth"][2]]
            dyaw=estimate[2]-truth[2]
            pairs.append({"sim_time":now,"slam":estimate,"truth_initial_frame":truth,
                          "error_m":math.dist(estimate[:2],truth[:2]),
                          "yaw_error_rad":abs(math.atan2(math.sin(dyaw),math.cos(dyaw)))})
        except Exception:missed+=1
    summary=json.loads((folder/"summary.json").read_text())
    if not (folder/"online_summary.json").exists():shutil.copy2(folder/"summary.json",folder/"online_summary.json")
    summary.update(
        slam_pose_samples=len(pairs),slam_pose_match_fraction=len(pairs)/len(samples),
        slam_position_rmse_m=float(np.sqrt(np.mean([p["error_m"]**2 for p in pairs]))) if pairs else None,
        slam_yaw_mae_rad=float(np.mean([p["yaw_error_rad"] for p in pairs])) if pairs else None,
        slam_evaluation_method="Offline interpolation of recorded estimated TF at evaluation timestamps; fixed initial frame, no fitted alignment.",
        slam_evaluation_time_range=[pairs[0]["sim_time"],pairs[-1]["sim_time"]] if pairs else [],
        slam_tf_counts=counts)
    (folder/"summary.json").write_text(json.dumps(summary,indent=2))
    (folder/"slam_pose_evaluation_offline.json").write_text(json.dumps(pairs))
    rclpy.shutdown()
    print(json.dumps({k:v for k,v in summary.items() if k not in ("probes","slam_tf_counts")},indent=2))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path)
    evaluate(p.parse_args().folder)
