"""Explicit degradation of Gazebo sensors; no ground-truth subscription."""
import argparse
from collections import deque
import copy
import heapq
import json
import math
from pathlib import Path
import time
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState,LaserScan,Imu,Image,CameraInfo
from nav_msgs.msg import Odometry
import yaml

from robot_sim.rover_control import integrate_encoders
from robot_sim.rover import MODEL_DIR,ROOT


class Sensors(Node):
    def __init__(self,args):
        super().__init__("realistic_sensors")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time",value=True)])
        self.profile=yaml.safe_load((ROOT/"robot_sim/config/mapping_sensors.yaml").read_text())
        if args.seed is not None:self.profile["seed"]=args.seed
        self.rng=np.random.default_rng(self.profile["seed"])
        self.config=json.loads((MODEL_DIR/"specification.json").read_text())
        self.queue=[];self.serial=0;self.previous=None;self.last_wheel=-1
        self.pose=(0.,0.,0.);self.velocities=deque(maxlen=4)
        self.bias=self.profile["imu"]["gyro_bias_rad_s"]
        self.previous_imu=None;self.info=None
        self.counts={};self.log=(args.output/"sensors.jsonl").open("w")
        self.publishers_by_name={
            "scan":self.create_publisher(LaserScan,"/scan",qos_profile_sensor_data),
            "imu":self.create_publisher(Imu,"/imu/data",qos_profile_sensor_data),
            "wheel":self.create_publisher(Odometry,"/wheel/odometry",10),
            "camera":self.create_publisher(Image,"/camera/image_raw",qos_profile_sensor_data)}
        self.info_pub=self.create_publisher(CameraInfo,"/camera/camera_info",10)
        for typ,topic,cb in [(LaserScan,"/sim/scan",self.scan),(Imu,"/sim/imu",self.imu),
                             (JointState,"/sim/joints",self.wheel),(Image,"/sim/camera/image",self.image),
                             (CameraInfo,"/sim/camera/camera_info",self.camera_info)]:
            self.create_subscription(typ,topic,cb,qos_profile_sensor_data)
        self.create_timer(.005,self.release)

    def enqueue(self,name,msg):
        p=self.profile["lidar" if name=="scan" else name]
        if self.rng.random()<p.get("message_dropout_probability",0):return
        latency=max(0,p["latency_s"]+self.rng.normal(0,p["jitter_s"]))
        self.serial+=1
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        heapq.heappush(self.queue,(stamp+latency,self.serial,name,msg))
        self.counts[name]=self.counts.get(name,0)+1

    def release(self):
        now=self.get_clock().now().nanoseconds/1e9
        while self.queue and self.queue[0][0]<=now:
            deadline,_,name,msg=heapq.heappop(self.queue)
            self.publishers_by_name[name].publish(msg)
            if name=="camera" and self.info:
                info=copy.deepcopy(self.info);info.header=msg.header;self.info_pub.publish(info)
        if self.serial%100==0:
            self.log.write(json.dumps({"sim_time":now,"counts":self.counts,"queued":len(self.queue)})+"\n")
            self.log.flush()

    def scan(self,msg):
        p=self.profile["lidar"]
        r=np.asarray(msg.ranges,dtype=float)
        valid=np.isfinite(r)&(r>=p["range_min_m"])&(r<=p["range_max_m"])
        r[np.isfinite(r)&(r>p["range_max_m"])]=np.inf
        r[np.isfinite(r)&(r<p["range_min_m"])]=np.nan
        r[valid]+=p["fixed_range_bias_m"]+self.rng.normal(0,p["white_std_m"]+p["range_std_fraction"]*r[valid])
        r[valid]=np.rint(r[valid]/p["resolution_m"])*p["resolution_m"]
        r[self.rng.random(len(r))<p["beam_dropout_probability"]]=np.nan
        outliers=self.rng.random(len(r))<p["outlier_probability"]
        r[outliers]=self.rng.uniform(p["range_min_m"],p["range_max_m"],outliers.sum())
        r[(r<p["range_min_m"])|(r>p["range_max_m"])&np.isfinite(r)]=np.nan
        msg.ranges=r.astype(np.float32).tolist()
        msg.range_min=p["range_min_m"];msg.range_max=p["range_max_m"]
        # Gazebo delivers instantaneous snapshots; do not claim per-ray timing/deskew.
        msg.scan_time=.1;msg.time_increment=0.
        msg.angle_min+=p["yaw_calibration_error_rad"];msg.angle_max+=p["yaw_calibration_error_rad"]
        self.enqueue("scan",msg)

    def imu(self,msg):
        p=self.profile["imu"]
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        dt=max(0,stamp-self.previous_imu) if self.previous_imu is not None else .02
        self.previous_imu=stamp
        self.bias+=self.rng.normal(0,p["gyro_bias_walk_std"]*math.sqrt(dt))
        for key in ("x","y","z"):
            value=getattr(msg.angular_velocity,key)+self.rng.normal(0,p["gyro_std_rad_s"])
            if key=="z":value+=self.bias
            setattr(msg.angular_velocity,key,value)
            setattr(msg.linear_acceleration,key,getattr(msg.linear_acceleration,key)+self.rng.normal(0,p["acceleration_std_m_s2"]))
        msg.orientation.x=msg.orientation.y=msg.orientation.z=0.;msg.orientation.w=1.
        msg.orientation_covariance[0]=-1. # No ideal world yaw leaked from simulator.
        for index in (0,4,8):
            msg.angular_velocity_covariance[index]=p["gyro_std_rad_s"]**2+.003**2
            msg.linear_acceleration_covariance[index]=p["acceleration_std_m_s2"]**2
        self.enqueue("imu",msg)

    def wheel(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if stamp-self.last_wheel<.019:return
        self.last_wheel=stamp
        p=self.profile["wheel"];names=("left_middle_joint","right_middle_joint")
        if not all(n in msg.name for n in names):return
        tick=2*math.pi/p["ticks_per_revolution"]
        angles=[round(msg.position[msg.name.index(n)]/tick)*tick for n in names]
        if self.previous is None:self.previous=(stamp,angles);return
        dt=stamp-self.previous[0]
        if dt<=0:return
        dl=(angles[0]-self.previous[1][0])*p["left_radius_scale"]
        dr=(angles[1]-self.previous[1][1])*p["right_radius_scale"]
        self.previous=(stamp,angles)
        radius=self.config["wheel_radius_m"];track=self.config["track_m"]*p["track_scale"]
        self.pose=integrate_encoders(self.pose,dl,dr,radius,track)
        self.velocities.append((radius*(dl+dr)/(2*dt),radius*(dr-dl)/(track*dt)))
        v,w=np.mean(self.velocities,axis=0)
        out=Odometry();out.header=msg.header;out.header.frame_id="odom";out.child_frame_id="base_footprint"
        out.pose.pose.position.x,out.pose.pose.position.y=self.pose[:2]
        out.pose.pose.orientation.z=math.sin(self.pose[2]/2);out.pose.pose.orientation.w=math.cos(self.pose[2]/2)
        out.twist.twist.linear.x=float(v);out.twist.twist.angular.z=float(w)
        for index,val in ((0,.015**2),(7,.03**2),(14,1e3),(21,1e3),(28,1e3),(35,.06**2)):
            out.twist.covariance[index]=val
            out.pose.covariance[index]=.01 if index in (0,7,35) else 1e3
        self.enqueue("wheel",out)

    def camera_info(self,msg):
        p=self.profile["camera"]
        for index in (0,4):msg.k[index]*=p["focal_calibration_scale"]
        self.info=msg

    def image(self,msg):
        p=self.profile["camera"]
        channels=3 if msg.encoding in ("rgb8","bgr8") else 4
        rgb=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
        rgb=rgb[:,:msg.width*channels].reshape(msg.height,msg.width,channels)[:,:,:3]
        if msg.encoding in ("bgr8","bgra8"):rgb=rgb[:,:,::-1]
        rgb=np.clip(rgb.astype(float)+self.rng.normal(0,p["image_std"],rgb.shape),0,255).astype(np.uint8)
        ok,jpeg=cv2.imencode(".jpg",rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,p["jpeg_quality"]])
        if ok:rgb=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)[:,:,::-1].copy()
        msg.encoding="rgb8";msg.step=msg.width*3;msg.data=rgb.tobytes()
        self.enqueue("camera",msg)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int)
    args=p.parse_args();rclpy.init();node=Sensors(args)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.log.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=="__main__":main()
