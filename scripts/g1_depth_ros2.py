#!/usr/bin/env python3
"""G1 RealSense ROS2 depth + calibrated TF -> timestamped PMT observations.

Run with the robot's ROS Python environment, not the Isaac conda environment.
No motor commands are published. See docs/DEPTH_PERCEPTION.md for TF contracts.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmt_tasks.depth_perception import CameraIntrinsics, LocalElevationMap
from pmt_tasks.realsense_perception import RobotSelfFilter, decode_depth_image, subsample_depth


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformListener, TransformException

    class DepthNode(Node):
        def __init__(self):
            super().__init__("pmt_g1_depth")
            defaults = {"depth_topic": "/camera/camera/depth/image_rect_raw", "info_topic": "/camera/camera/depth/camera_info",
                        "odom_frame": "odom", "torso_frame": "torso_link", "pixel_stride": 4,
                        "map_max_age": 2., "max_input_age": .2, "publish_hz": 50., "self_padding": .04,
                        "urdf": str(Path(__file__).resolve().parents[1] / "pmt_tasks/assets/unitree_description/urdf/g1/main.urdf")}
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            self.cfg = {key: self.get_parameter(key).value for key in defaults}
            if self.cfg["pixel_stride"] < 1 or min(self.cfg["publish_hz"], self.cfg["max_input_age"]) <= 0:
                raise ValueError("Rates, timeout and pixel stride must be positive")
            self.mapper = LocalElevationMap(max_age_s=self.cfg["map_max_age"])
            self.self_filter = RobotSelfFilter(self.cfg["urdf"], self.cfg["self_padding"])
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)
            self.info = None
            self.last_depth = -np.inf
            self.create_subscription(CameraInfo, self.cfg["info_topic"], self.on_info, qos_profile_sensor_data)
            self.create_subscription(Image, self.cfg["depth_topic"], self.on_depth, qos_profile_sensor_data)
            self.publisher = self.create_publisher(String, "/pmt/depth_observation", 1)
            self.create_timer(1 / self.cfg["publish_hz"], self.publish_observation)
            self.create_service(Empty, "~/reset_map", self.reset_map)
            self.get_logger().info("Waiting for rectified depth, CameraInfo and synchronized odom/robot/camera TF")

        def reset_map(self, request, response):
            self.mapper.reset()
            self.last_depth = -np.inf
            return response

        def on_info(self, info):
            self.info = info

        def transform(self, frame, stamp):
            message = self.tf.lookup_transform(self.cfg["odom_frame"], frame, stamp)
            t, q = message.transform.translation, message.transform.rotation
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            pose[:3, 3] = [t.x, t.y, t.z]
            return pose, Time.from_msg(message.header.stamp).nanoseconds / 1e9

        def on_depth(self, image):
            if self.info is None:
                return
            try:
                info = self.info
                if image.header.frame_id != info.header.frame_id or (image.width, image.height) != (info.width, info.height):
                    raise ValueError("Depth and CameraInfo frames/dimensions differ")
                stamp = Time.from_msg(image.header.stamp)
                seconds = stamp.nanoseconds / 1e9
                age = self.get_clock().now().nanoseconds / 1e9 - seconds
                if not 0 <= age <= self.cfg["max_input_age"]:
                    raise ValueError("Depth timestamp is stale or ahead of ROS clock")
                # P describes the rectified image; do not use RGB intrinsics.
                p = np.asarray(info.p).reshape(3, 4)
                k = CameraIntrinsics(info.width, info.height, p[0, 0], p[1, 1], p[0, 2], p[1, 2])
                depth = decode_depth_image(image.data, image.width, image.height, image.step, image.encoding, bool(image.is_bigendian))
                depth, k = subsample_depth(depth, k, self.cfg["pixel_stride"])
                camera, _ = self.transform(image.header.frame_id, stamp)
                torso, _ = self.transform(self.cfg["torso_frame"], stamp)
                links = {name: self.transform(name, stamp)[0] for name in self.self_filter.link_names}
                mask = self.self_filter.valid_mask(depth, k, camera, links)
                self.mapper.integrate(depth, k, camera, seconds, torso[:2, 3], mask)
                self.last_depth = seconds
            except (ValueError, KeyError, TransformException) as error:
                self.get_logger().warning(f"Depth frame rejected: {error}", throttle_duration_sec=2.)

        def publish_observation(self):
            if not np.isfinite(self.last_depth):
                return
            try:
                now = self.get_clock().now().nanoseconds / 1e9
                torso, stamp = self.transform(self.cfg["torso_frame"], Time())
                if not 0 <= now - stamp <= self.cfg["max_input_age"] or stamp < self.last_depth:
                    raise ValueError("Torso TF is stale or precedes latest depth")
                yaw = np.arctan2(torso[1, 0], torso[0, 0])
                scan = self.mapper.sample(torso[:3, 3], yaw, now)
                fresh = 0 <= now - self.last_depth <= self.cfg["max_input_age"]
                if not fresh:
                    scan["policy_valid"] = [False] * 187
                    scan["policy_height"] = [-3.] * 187
                    scan["hit_valid"] = [False] * 187
                    scan["hits_w"] = [[0., 0., 0.]] * 187
                    scan["age_s"] = [-1.] * 187
                message = String()
                message.data = json.dumps({"timestamp": now, "pose_timestamp": stamp,
                                           "depth_timestamp": self.last_depth, "sensor_fresh": fresh,
                                           "vision": scan["policy_height"] + [float(v) for v in scan["policy_valid"]],
                                           "height_scan": scan}, allow_nan=False)
                self.publisher.publish(message)
            except (ValueError, TransformException) as error:
                self.get_logger().warning(f"Observation withheld: {error}", throttle_duration_sec=2.)

    rclpy.init()
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
