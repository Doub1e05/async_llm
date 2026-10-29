"""Scripted mapping route and separate privileged evaluator. No LLM."""
import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from geometry_msgs.msg import Twist,PoseStamped
from nav_msgs.msg import Odometry,OccupancyGrid
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from ros_gz_interfaces.msg import Contacts
from tf2_ros import Buffer,TransformListener
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

from robot_sim.control import wrap_angle


class Experiment(Node):
    def __init__(self,args):
        super().__init__("mapping_evaluator")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time",value=True)])
        self.args=args;self.odom=self.truth=self.wheel=None
        self.map=None;self.local=None;self.scan=None;self.contacts=0
        self.started=None;self.phase="startup";self.reached=0;self.samples=[];self.probes=[]
        self.slam_pairs=[];self.slam_index=0
        self.last_probe=-1;self.last_actor=-1;self.actor_requests=0;self.actor_failures=0
        self.pose_pub=self.create_publisher(Twist,"/cmd_vel",1)
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        for typ,topic,cb,qos in [
            (Odometry,"/odom",self.odometry,10),(Odometry,"/wheel/odometry",self.wheels,10),
            (PoseStamped,"/evaluation/ground_truth",self.ground_truth,qos_profile_sensor_data),
            (LaserScan,"/scan",self.laser,qos_profile_sensor_data),
            (OccupancyGrid,"/map",lambda m:setattr(self,"map",m),latched),
            (OccupancyGrid,"/mapping/dynamic_grid",lambda m:setattr(self,"local",m),latched),
            (Contacts,"/contacts",self.contact,qos_profile_sensor_data)]:
            self.create_subscription(typ,topic,cb,qos)
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.gz=GzNode();self.log=(args.output/"evaluation_trajectory.jsonl").open("w")

    def odometry(self,msg):self.odom=msg
    def wheels(self,msg):self.wheel=msg
    def laser(self,msg):self.scan=msg
    def contact(self,msg):
        if msg.contacts:self.contacts+=1
    def ground_truth(self,msg):
        self.truth=msg
        if self.odom:
            now=self.get_clock().now().nanoseconds/1e9
            q=self.odom.pose.pose.orientation;p=self.odom.pose.pose.position
            truth=msg.pose.position
            t=msg.pose.orientation
            record={"sim_time":now,"phase":self.phase,"truth":[truth.x,truth.y,2*math.atan2(t.z,t.w)],
                    "odom":[p.x,p.y,2*math.atan2(q.z,q.w)],
                    "wheel":[self.wheel.pose.pose.position.x,self.wheel.pose.pose.position.y,
                             2*math.atan2(self.wheel.pose.pose.orientation.z,self.wheel.pose.pose.orientation.w)] if self.wheel else None}
            self.samples.append(record);self.log.write(json.dumps(record)+"\n")

    def send(self,v,w):
        msg=Twist();msg.linear.x=float(v);msg.angular.z=float(w);self.pose_pub.publish(msg)

    def actor(self,elapsed):
        if elapsed-self.last_actor<.15:return
        self.last_actor=elapsed
        y=min(-.65,-2.8+max(0,elapsed-5)*.25)
        p=Pose();p.name="moving_box";p.position.x=-1.8;p.position.y=y;p.position.z=.225;p.orientation.w=1.
        ok,response=self.gz.request("/world/burger_course/set_pose",p,Pose,Boolean,200)
        self.actor_requests+=1
        if not ok or not response.data:self.actor_failures+=1

    def slam_evaluation(self,now):
        # Delayed evaluation ensures timestamped TF has arrived. Truth is used
        # only here; mapper, SLAM and EKF remain sensor-only.
        while self.slam_index<len(self.samples):
            sample=self.samples[self.slam_index]
            if sample["sim_time"]>now-.15:break
            self.slam_index+=1
            try:
                stamp=Time(seconds=int(sample["sim_time"]),nanoseconds=int((sample["sim_time"]%1)*1e9))
                tf=self.buffer.lookup_transform("map","base_footprint",stamp)
                p=tf.transform.translation;q=tf.transform.rotation
                estimate=[p.x,p.y,2*math.atan2(q.z,q.w)]
                truth=[sample["truth"][0]+4,sample["truth"][1]+2.8,sample["truth"][2]]
                self.slam_pairs.append({"sim_time":sample["sim_time"],"slam":estimate,"truth_initial_frame":truth,
                                        "error_m":math.dist(estimate[:2],truth[:2]),
                                        "yaw_error_rad":abs(wrap_angle(estimate[2]-truth[2]))})
            except Exception:pass

    def probe(self,elapsed):
        if elapsed-self.last_probe<.5 or self.local is None:return
        self.last_probe=elapsed
        grid=self.local;values=np.array(grid.data).reshape(grid.info.height,grid.info.width)
        x=2.075;y=0.0 # Initial front face of test obstacle in initial-odom coordinates.
        column=int((x-grid.info.origin.position.x)/grid.info.resolution)
        row=int((y-grid.info.origin.position.y)/grid.info.resolution)
        patch=values[max(0,row-2):row+3,max(0,column-2):column+3]
        self.probes.append({"elapsed":elapsed,"max":int(patch.max()),"mean":float(patch.mean()),
                            "occupied":int(np.sum(patch>=65)),"unknown":int(np.sum(patch<0))})

    def run(self):
        deadline=time.monotonic()+150
        while time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.03)
            if self.odom and self.truth and self.map and self.local and self.scan:break
        else:raise TimeoutError("Mapping readiness: "+str([self.odom is not None,self.truth is not None,self.map is not None,self.local is not None,self.scan is not None]))
        self.started=self.get_clock().now().nanoseconds/1e9
        goals=[(3.0,0.0),(3.0,1.6),(0.0,1.6),(0.0,0.0)]
        wall_end=time.monotonic()+self.args.duration*12
        while rclpy.ok():
            now=self.get_clock().now().nanoseconds/1e9;elapsed=now-self.started
            if elapsed>=self.args.duration or self.reached==len(goals):break
            if time.monotonic()>wall_end:raise TimeoutError("Simulation clock stalled")
            self.actor(elapsed);self.probe(elapsed);self.slam_evaluation(now)
            v=w=0.
            if elapsed>=5:
                p=self.odom.pose.pose.position;q=self.odom.pose.pose.orientation
                yaw=2*math.atan2(q.z,q.w);gx,gy=goals[self.reached]
                distance=math.hypot(gx-p.x,gy-p.y)
                heading=wrap_angle(math.atan2(gy-p.y,gx-p.x)-yaw)
                if distance<.09:
                    self.reached+=1;self.phase="waypoint_"+str(self.reached)
                else:
                    w=max(-.8,min(.8,2.2*heading))
                    v=min(.13,.7*distance) if abs(heading)<.30 else 0.
                    if self.scan:
                        front=[r for i,r in enumerate(self.scan.ranges) if
                               abs(self.scan.angle_min+i*self.scan.angle_increment)<.35 and math.isfinite(r)]
                        if front and min(front)<.30:v=0.
                    self.phase="drive_"+str(self.reached)
            self.send(v,w)
            rclpy.spin_once(self,timeout_sec=.02)
            time.sleep(.01)
        self.send(0,0)
        end=self.get_clock().now().nanoseconds/1e9+3
        while self.get_clock().now().nanoseconds/1e9<end:
            self.send(0,0);rclpy.spin_once(self,timeout_sec=.03);time.sleep(.01)
        self.slam_evaluation(self.get_clock().now().nanoseconds/1e9)
        self.save()

    def save(self):
        s=self.samples
        errors=[math.hypot(p["odom"][0]-(p["truth"][0]+4),p["odom"][1]-(p["truth"][1]+2.8)) for p in s]
        wheel_errors=[math.hypot(p["wheel"][0]-(p["truth"][0]+4),p["wheel"][1]-(p["truth"][1]+2.8)) for p in s if p["wheel"]]
        yaw_errors=[abs(wrap_angle(p["odom"][2]-p["truth"][2])) for p in s]
        initial=[p for p in self.probes if 1<p["elapsed"]<4.8]
        after=[p for p in self.probes if 15<p["elapsed"]<24]
        dynamic_seen=any(p["occupied"]>0 for p in initial)
        dynamic_cleared=bool(after) and any(p["occupied"]==0 for p in after)
        occupied=np.asarray(self.map.data).reshape(self.map.info.height,self.map.info.width)
        pairs=self.slam_pairs
        (self.args.output/"slam_pose_evaluation.json").write_text(json.dumps(pairs))
        # Independent comparison against static collision surfaces, never an input to SLAM.
        surfaces=[]
        world=ET.parse(self.args.output/"world.sdf")
        for model in world.findall("./world/model"):
            if model.get("name") in {"ground","yandex_rover","moving_box"}:continue
            pose=[float(x) for x in model.findtext("pose","0 0 0 0 0 0").split()]
            box=model.find(".//collision/geometry/box/size")
            if box is None:continue
            size=[float(x) for x in box.text.split()]
            if not (pose[2]-size[2]/2<=.162124<=pose[2]+size[2]/2):continue
            surfaces.append((pose[0]-size[0]/2,pose[0]+size[0]/2,pose[1]-size[1]/2,pose[1]+size[1]/2))
        rows,cols=np.nonzero(occupied>=65)
        x=self.map.info.origin.position.x+(cols+.5)*self.map.info.resolution-4
        y=self.map.info.origin.position.y+(rows+.5)*self.map.info.resolution-2.8
        # Exclude the swept actor area from a static-map accuracy score.
        keep=~((x>-2.05)&(x<-1.55)&(y>-3.1)&(y<-.4))
        errors_to_surface=[]
        for lo_x,hi_x,lo_y,hi_y in surfaces:
            dx=np.maximum(np.maximum(lo_x-x,x-hi_x),0)
            dy=np.maximum(np.maximum(lo_y-y,y-hi_y),0)
            outside=np.hypot(dx,dy)
            inside=(dx==0)&(dy==0)
            outside[inside]=np.minimum.reduce([x[inside]-lo_x,hi_x-x[inside],y[inside]-lo_y,hi_y-y[inside]])
            errors_to_surface.append(outside)
        surface_distance=np.min(errors_to_surface,axis=0)[keep] if errors_to_surface else np.array([])
        summary={"route_completed":self.reached==4,"waypoints_reached":self.reached,
                 "contacts":self.contacts,"actor_requests":self.actor_requests,"actor_failures":self.actor_failures,
                 "position_rmse_m":float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                 "wheel_position_rmse_m":float(np.sqrt(np.mean(np.square(wheel_errors)))) if wheel_errors else None,
                 "yaw_mae_rad":float(np.mean(yaw_errors)) if yaw_errors else None,
                 "slam_pose_samples":len(pairs),
                 "slam_position_rmse_m":float(np.sqrt(np.mean([p["error_m"]**2 for p in pairs]))) if pairs else None,
                 "slam_yaw_mae_rad":float(np.mean([p["yaw_error_rad"] for p in pairs])) if pairs else None,
                 "static_surface_median_m":float(np.median(surface_distance)) if len(surface_distance) else None,
                 "static_surface_p95_m":float(np.quantile(surface_distance,.95)) if len(surface_distance) else None,
                 "map_known_area_m2":float(np.sum(occupied>=0)*self.map.info.resolution**2),
                 "map_occupied_cells":int(np.sum(occupied>=65)),
                 "dynamic_obstacle_seen":dynamic_seen,"dynamic_obstacle_cleared":dynamic_cleared,
                 "probes":self.probes,
                 "ground_truth_role":"evaluation only; never fed to EKF, SLAM or mapper"}
        summary["passed"]=bool(self.contacts==0 and self.reached==4 and dynamic_seen and dynamic_cleared
                                and summary["map_occupied_cells"]>50 and self.actor_failures==0)
        (self.args.output/"summary.json").write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k!="probes"}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--duration",type=float,default=150)
    args=p.parse_args();rclpy.init();node=Experiment(args)
    try:node.run()
    except Exception as e:
        (args.output/"experiment_error.txt").write_text(str(e));raise
    finally:
        if rclpy.ok():node.send(0,0)
        node.log.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if json.loads((args.output/"summary.json").read_text())["passed"] else 2

if __name__=="__main__":raise SystemExit(main())
