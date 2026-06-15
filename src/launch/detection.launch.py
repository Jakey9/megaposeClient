from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_name", default_value="megapose-1.0-RGB"),
            DeclareLaunchArgument("mesh_dir", default_value=""),
            DeclareLaunchArgument("mesh_units", default_value="mm"),
            DeclareLaunchArgument("yolo_model", default_value="yolov8n.pt"),
            DeclareLaunchArgument("yolo_confidence", default_value="0.5"),
            DeclareLaunchArgument("target_label", default_value=""),
            DeclareLaunchArgument("object_label", default_value=""),
            DeclareLaunchArgument("pose_score_threshold", default_value="0.3"),
            DeclareLaunchArgument(
                "rgb_topic", default_value="/camera/camera/color/image_raw"
            ),
            DeclareLaunchArgument(
                "depth_topic", default_value="/camera/camera/depth/image_rect_raw"
            ),
            DeclareLaunchArgument(
                "info_topic", default_value="/camera/camera/color/camera_info"
            ),
            # MegaPose tuning (-1 = use model default)
            DeclareLaunchArgument("n_refiner_iterations", default_value="-1"),
            DeclareLaunchArgument("n_pose_hypotheses", default_value="-1"),
            DeclareLaunchArgument("bsz_images", default_value="128"),
            DeclareLaunchArgument("bsz_objects", default_value="8"),
            # Overlay
            DeclareLaunchArgument("publish_overlay", default_value="true"),
            Node(
                package="detection_client",
                executable="detection_node",
                name="detection_node",
                output="screen",
                parameters=[
                    {
                        "model_name": LaunchConfiguration("model_name"),
                        "mesh_dir": LaunchConfiguration("mesh_dir"),
                        "mesh_units": LaunchConfiguration("mesh_units"),
                        "yolo_model": LaunchConfiguration("yolo_model"),
                        "yolo_confidence": LaunchConfiguration("yolo_confidence"),
                        "target_label": LaunchConfiguration("target_label"),
                        "object_label": LaunchConfiguration("object_label"),
                        "pose_score_threshold": LaunchConfiguration(
                            "pose_score_threshold"
                        ),
                        "rgb_topic": LaunchConfiguration("rgb_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "info_topic": LaunchConfiguration("info_topic"),
                        "n_refiner_iterations": LaunchConfiguration(
                            "n_refiner_iterations"
                        ),
                        "n_pose_hypotheses": LaunchConfiguration("n_pose_hypotheses"),
                        "bsz_images": LaunchConfiguration("bsz_images"),
                        "bsz_objects": LaunchConfiguration("bsz_objects"),
                        "publish_overlay": LaunchConfiguration("publish_overlay"),
                    }
                ],
            ),
        ]
    )
