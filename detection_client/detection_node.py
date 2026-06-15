"""ROS2 node: YOLO detection -> MegaPose6D 6-DOF pose estimation -> TF + Marker."""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import rclpy
import torch
import trimesh
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from ultralytics import YOLO
from visualization_msgs.msg import Marker

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.tensor_collection import PandasTensorCollection


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        self._declare_params()

        model_name = self.get_parameter("model_name").value
        mesh_dir = self.get_parameter("mesh_dir").value
        mesh_units = self.get_parameter("mesh_units").value
        yolo_model = self.get_parameter("yolo_model").value
        self.yolo_confidence = self.get_parameter("yolo_confidence").value
        self.target_label = self.get_parameter("target_label").value
        object_label_param = self.get_parameter("object_label").value
        self.pose_score_threshold = self.get_parameter("pose_score_threshold").value
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value
        n_refiner_iterations = self.get_parameter("n_refiner_iterations").value
        n_pose_hypotheses = self.get_parameter("n_pose_hypotheses").value
        bsz_images = self.get_parameter("bsz_images").value
        bsz_objects = self.get_parameter("bsz_objects").value
        self.publish_overlay = self.get_parameter("publish_overlay").value

        if not mesh_dir:
            mesh_dir = str(Path(__file__).resolve().parent.parent / "meshes")
        mesh_dir = Path(mesh_dir)

        model_info = NAMED_MODELS[model_name]
        self.requires_depth = model_info["requires_depth"]
        self.inference_params = dict(model_info["inference_parameters"])
        if n_refiner_iterations > 0:
            self.inference_params["n_refiner_iterations"] = n_refiner_iterations
        if n_pose_hypotheses > 0:
            self.inference_params["n_pose_hypotheses"] = n_pose_hypotheses

        # -- Load YOLO --
        self.get_logger().info(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)

        # -- Build object dataset from meshes/<label>/*.ply|obj --
        self.get_logger().info(f"Loading meshes from: {mesh_dir}")
        if object_label_param:
            self.object_label = object_label_param
        else:
            self.object_label = self._auto_detect_label(mesh_dir)
        object_dir = mesh_dir / self.object_label
        mesh_path = self._find_mesh(object_dir)
        rigid_objects = [
            RigidObject(label=self.object_label, mesh_path=mesh_path, mesh_units=mesh_units)
        ]
        self.object_dataset = RigidObjectDataset(rigid_objects)
        self.get_logger().info(f"Object label: '{self.object_label}', mesh: {mesh_path}")

        # -- Load mesh geometry for contour overlay --
        if self.publish_overlay:
            self._load_overlay_mesh(mesh_path, mesh_units)

        # -- Load MegaPose (heavy, one-time) --
        self.get_logger().info(f"Loading MegaPose model: {model_name} (this may take a while)")
        self.pose_estimator = load_named_model(
            model_name, self.object_dataset, bsz_images=bsz_images
        ).cuda()
        self.pose_estimator.bsz_objects = bsz_objects
        self.get_logger().info("MegaPose model loaded.")
        self.get_logger().info(
            f"Inference params: {self.inference_params}, "
            f"bsz_images={bsz_images}, bsz_objects={bsz_objects}"
        )

        # -- State --
        self.current_pose: Optional[PoseEstimatesType] = None
        self.bridge = CvBridge()

        # -- Publishers --
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(Marker, "~/object_marker", 10)
        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(Image, "~/overlay", 10)

        # -- Subscribers (time-synced) --
        self.sub_rgb = Subscriber(self, Image, rgb_topic)
        self.sub_info = Subscriber(self, CameraInfo, info_topic)

        if self.requires_depth:
            self.sub_depth = Subscriber(self, Image, depth_topic)
            self.sync = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_depth, self.sub_info], queue_size=5, slop=0.05
            )
            self.sync.registerCallback(self._callback_rgbd)
        else:
            self.sync = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_info], queue_size=5, slop=0.05
            )
            self.sync.registerCallback(self._callback_rgb)

        self.get_logger().info("Detection node ready. Waiting for camera frames...")

    def _declare_params(self):
        self.declare_parameter("model_name", "megapose-1.0-RGB")
        self.declare_parameter("mesh_dir", "")
        self.declare_parameter("mesh_units", "mm")
        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("yolo_confidence", 0.5)
        self.declare_parameter("target_label", "")
        self.declare_parameter("object_label", "")
        self.declare_parameter("pose_score_threshold", 0.3)
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("info_topic", "/camera/camera/color/camera_info")
        # MegaPose tuning (-1 = use model default)
        self.declare_parameter("n_refiner_iterations", -1)
        self.declare_parameter("n_pose_hypotheses", -1)
        self.declare_parameter("bsz_images", 128)
        self.declare_parameter("bsz_objects", 8)
        # Overlay
        self.declare_parameter("publish_overlay", True)

    # -- Mesh helpers --

    @staticmethod
    def _auto_detect_label(mesh_dir: Path) -> str:
        """Pick the first subdirectory in mesh_dir as the object label."""
        for d in sorted(mesh_dir.iterdir()):
            if d.is_dir():
                return d.name
        raise FileNotFoundError(
            f"No object subdirectory found in {mesh_dir}. "
            "Place your mesh at meshes/<label>/model.ply"
        )

    @staticmethod
    def _find_mesh(object_dir: Path) -> Path:
        for f in object_dir.iterdir():
            if f.suffix in {".obj", ".ply"}:
                return f
        raise FileNotFoundError(f"No .obj or .ply mesh found in {object_dir}")

    def _load_overlay_mesh(self, mesh_path: Path, mesh_units: str):
        """Load mesh vertices/faces for lightweight contour projection."""
        mesh = trimesh.load(str(mesh_path), force="mesh")
        scale = {"mm": 0.001, "m": 1.0}[mesh_units]
        self.overlay_vertices = np.array(mesh.vertices, dtype=np.float64) * scale
        self.overlay_faces = np.array(mesh.faces, dtype=np.int32)
        self.get_logger().info(
            f"Overlay mesh loaded: {len(self.overlay_vertices)} verts, "
            f"{len(self.overlay_faces)} faces"
        )

    # -- Subscriber callbacks --

    def _callback_rgb(self, rgb_msg: Image, info_msg: CameraInfo):
        self._process(rgb_msg, None, info_msg)

    def _callback_rgbd(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self._process(rgb_msg, depth_msg, info_msg)

    # -- Main processing loop --

    def _process(self, rgb_msg: Image, depth_msg: Optional[Image], info_msg: CameraInfo):
        t_start = time.perf_counter()

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        K = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)

        depth = None
        if depth_msg is not None:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth = depth_raw.astype(np.float32) / 1000.0

        observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()

        if self.current_pose is None:
            bbox = self._detect_yolo(rgb)
            if bbox is None:
                return
            detections = self._build_detections(bbox)
            self.get_logger().info("Running MegaPose full pipeline (coarse + refiner)...")
            output, extra = self.pose_estimator.run_inference_pipeline(
                observation, detections=detections, **self.inference_params
            )
            self.current_pose = output
            self.get_logger().info(f"Init done. {extra.get('timing_str', '')}")
        else:
            output, extra = self.pose_estimator.run_inference_pipeline(
                observation, coarse_estimates=self.current_pose, **self.inference_params
            )
            score = float(output.infos["pose_score"].iloc[0])
            if score < self.pose_score_threshold:
                self.get_logger().warn(
                    f"Pose score {score:.3f} below threshold, re-initialising..."
                )
                self.current_pose = None
                return
            self.current_pose = output

        pose_4x4 = output.poses[0].cpu().numpy()
        self._publish_tf(pose_4x4, rgb_msg.header)
        self._publish_marker(pose_4x4, rgb_msg.header)

        if self.publish_overlay:
            self._publish_overlay(rgb, pose_4x4, K, rgb_msg.header)

        elapsed = time.perf_counter() - t_start
        self.get_logger().info(f"Frame: {elapsed:.3f}s ({1.0 / max(elapsed, 1e-9):.1f} FPS)")

    # -- YOLO --

    def _detect_yolo(self, rgb: np.ndarray) -> Optional[list]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self.yolo(bgr, conf=self.yolo_confidence, verbose=False)
        best_box = None
        best_conf = 0.0
        best_cls = ""
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                conf = float(box.conf)
                matches = not self.target_label or cls_name == self.target_label
                if matches and conf > best_conf:
                    best_conf = conf
                    best_cls = cls_name
                    best_box = box.xyxy[0].cpu().numpy().tolist()
        if best_box is None:
            label_str = f"'{self.target_label}'" if self.target_label else "any object"
            self.get_logger().warn(f"YOLO: {label_str} not detected, skipping frame.")
        else:
            self.get_logger().info(
                f"YOLO: '{best_cls}' bbox={[f'{v:.0f}' for v in best_box]} "
                f"conf={best_conf:.2f}"
            )
        return best_box

    # -- Detection conversion --

    def _build_detections(self, bbox: list) -> DetectionsType:
        infos = pd.DataFrame(
            dict(label=[self.object_label], batch_im_id=[0], instance_id=[0])
        )
        bboxes = torch.tensor([bbox], dtype=torch.float32)
        return PandasTensorCollection(infos=infos, bboxes=bboxes).cuda()

    # -- TF --

    def _publish_tf(self, pose: np.ndarray, header):
        t = TransformStamped()
        t.header = header
        t.child_frame_id = "detected_object"

        t.transform.translation.x = float(pose[0, 3])
        t.transform.translation.y = float(pose[1, 3])
        t.transform.translation.z = float(pose[2, 3])

        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)

    # -- Marker --

    def _publish_marker(self, pose: np.ndarray, header):
        m = Marker()
        m.header = header
        m.ns = "megapose"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = float(pose[0, 3])
        m.pose.position.y = float(pose[1, 3])
        m.pose.position.z = float(pose[2, 3])

        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        m.pose.orientation.x = float(q[0])
        m.pose.orientation.y = float(q[1])
        m.pose.orientation.z = float(q[2])
        m.pose.orientation.w = float(q[3])

        m.scale.x = 0.05
        m.scale.y = 0.05
        m.scale.z = 0.05

        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8

        m.lifetime.sec = 0

        self.marker_pub.publish(m)

    # -- Contour overlay --

    def _publish_overlay(self, rgb: np.ndarray, pose: np.ndarray, K: np.ndarray, header):
        """Project mesh silhouette onto the RGB frame and publish as image."""
        h, w = rgb.shape[:2]
        R, t = pose[:3, :3], pose[:3, 3]

        v_cam = (R @ self.overlay_vertices.T + t.reshape(3, 1))  # (3, N)
        in_front = v_cam[2] > 0
        if not np.any(in_front):
            return

        v_proj = K @ v_cam  # (3, N)
        uv = (v_proj[:2] / v_proj[2:]).T  # (N, 2)

        mask = np.zeros((h, w), dtype=np.uint8)
        face_pts = uv[self.overlay_faces].astype(np.int32)  # (F, 3, 2)
        cv2.fillPoly(mask, face_pts, 255)

        canny = cv2.Canny(mask, 30, 100)
        kernel = np.ones((3, 3), dtype=np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=1)

        overlay = rgb.copy()
        overlay[canny > 0] = (0, 255, 0)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        overlay_msg.header = header
        self.overlay_pub.publish(overlay_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
