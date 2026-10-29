import math
import unittest
from pathlib import Path
import numpy as np
import yaml
from robot_sim.mapping.core import LocalGrid,filter_scan,project,transform_points,ClusterTracker,occupancy_to_pgm


class MappingTests(unittest.TestCase):
    def cell(self,grid,point,stamp):
        index,_=grid.index(np.array([point]))
        x,y=index[0]
        return int(grid.occupancy(stamp)[y,x])

    def test_rays_do_not_clear_behind_obstacle(self):
        grid=LocalGrid(resolution=.1,size=4)
        grid.update([0,0],[[1,0]],[True],1)
        self.assertGreaterEqual(self.cell(grid,[1,0],1),65)
        self.assertLess(self.cell(grid,[.5,0],1),50)
        self.assertEqual(self.cell(grid,[1.5,0],1),-1)

    def test_unseen_expiration_is_unknown_not_free(self):
        grid=LocalGrid(resolution=.1,size=4,ttl=2)
        grid.update([0,0],[[1,0]],[True],1)
        self.assertEqual(self.cell(grid,[1,0],3.1),-1)
        grid.update([0,0],[[1,0]],[True],4)
        self.assertEqual(self.cell(grid,[1,0],4),70)

    def test_reobservation_clears_moved_obstacle(self):
        grid=LocalGrid(resolution=.1,size=4)
        grid.update([0,0],[[1,0]],[True],1)
        for i in range(5):grid.update([0,0],[[1.8,0]],[False],1.1+i*.1)
        self.assertLess(self.cell(grid,[1,0],2),35)
        self.assertEqual(self.cell(grid,[1.9,0],2),-1)

    def test_beam_density_does_not_duplicate_confidence(self):
        a=LocalGrid(.1,4);b=LocalGrid(.1,4)
        a.update([0,0],[[1,0]],[True],1)
        b.update([0,0],[[1,0]]*20,[True]*20,1)
        np.testing.assert_array_equal(a.odds,b.odds)

    def test_rgb_projection_needs_measured_depth(self):
        k=[100,0,160,0,100,120,0,0,1]
        points=np.array([[0,0,1],[1,0,2],[0,0,-1],[10,0,1]])
        uv,valid=project(points,k,320,240)
        np.testing.assert_array_equal(valid,[True,True,False,False])
        np.testing.assert_allclose(uv[:2],[[160,120],[210,120]])

    def test_rigid_frame_rotation(self):
        p=transform_points([[1,0,0]],[2,3,0],[0,0,math.sin(math.pi/4),math.cos(math.pi/4)])
        np.testing.assert_allclose(p,[[2,4,0]],atol=1e-8)

    def test_missing_scan_beams_do_not_become_free_measurements(self):
        values=filter_scan([np.nan,1.,1.,.1,1.,1.,np.inf])
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[3]))
        self.assertTrue(np.isposinf(values[-1]))

    def test_ekf_does_not_fuse_correlated_encoder_pose_or_simulator_yaw(self):
        root=Path(__file__).resolve().parents[1]
        cfg=yaml.safe_load((root/"config/mapping_ekf.yaml").read_text())["ekf_filter_node"]["ros__parameters"]
        self.assertFalse(any(cfg["odom0_config"][:6]))
        self.assertEqual(np.flatnonzero(cfg["imu0_config"]).tolist(),[11])
        self.assertTrue(cfg["two_d_mode"])
        self.assertTrue(cfg["publish_tf"])

    def test_map_export_preserves_unknown_under_yaml_thresholds(self):
        pixels=occupancy_to_pgm([-1,0,20,50,90,100])
        probabilities=1-pixels.astype(float)/255
        restored=np.where(probabilities>.65,100,np.where(probabilities<.35,0,-1))
        np.testing.assert_array_equal(restored,[-1,0,0,-1,100,100])

    def test_static_track_is_not_classified_as_moving(self):
        tracker=ClusterTracker()
        for index in range(15):
            tracks=tracker.update([[1+(.001 if index%2 else 0),2]],index*.2)
        self.assertFalse(any(t["moving"] for t in tracks))


if __name__=="__main__":unittest.main()
