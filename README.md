# MegaPose Client

ROS2 node for real-time 6-DOF object pose estimation using [MegaPose6D](https://github.com/megapose6d/megapose6d) with YOLO-based object detection. Subscribes to a RealSense D455 camera, detects objects, estimates their 6D pose, and publishes results as TF transforms and RViz markers.

## How It Works

```
RealSense D455 ─── RGB + Depth + CameraInfo
                        │
                        ▼
                   YOLO Detection ──── 2D bounding box [xmin, ymin, xmax, ymax]
                        │
                        ▼
                MegaPose6D Inference ── 4×4 object-to-camera pose (SE3)
                        │
                  ┌─────┴─────┐
                  ▼           ▼
              TF Broadcast   RViz Marker
```

The node operates in two phases:

1. **Initialisation** -- YOLO detects the target object and produces a bounding box. MegaPose runs the full pipeline (coarse SO(3) grid search + refiner). This is slow (~1-2s) but only happens once.
2. **Tracking** -- Subsequent frames skip detection and coarse estimation entirely. The previous pose is passed directly to MegaPose's refiner for fast iterative updates (~5-10 FPS). If the pose confidence drops below a threshold, the node automatically re-initialises via YOLO.

## Prerequisites

- **ROS2** (Humble or later)
- **MegaPose6D** installed as a Python package:
  ```bash
  cd /path/to/megapose6d
  pip install -e .
  ```
- **MegaPose model weights** downloaded to `megapose6d/local_data/megapose-models/`:
  ```bash
  python -m megapose.scripts.download --models
  ```
- **Ultralytics** (YOLO):
  ```bash
  pip install ultralytics
  ```
- **RealSense ROS2 driver**:
  ```bash
  sudo apt install ros-${ROS_DISTRO}-realsense2-camera
  ```
- **NVIDIA GPU** with CUDA (required by MegaPose and YOLO)

## Setup

### 1. Prepare your object mesh

Place a 3D mesh (`.ply` or `.obj`) for your target object in the `meshes/` directory:

```
meshes/
  └── <object_label>/
        └── model.ply    # or model.obj
```

The `<object_label>` folder name must match the `object_label` parameter (defaults to `target_label` if not set).

### 2. Build the package

```bash
cd <your_ros2_workspace>/src
ln -s /path/to/megaposeClient .
cd ..
colcon build --packages-select detection_client
source install/setup.bash
```

### 3. Start the RealSense camera

```bash
ros2 launch realsense2_camera rs_launch.py
```

### 4. Launch the node

```bash
ros2 launch detection_client detection.launch.py target_label:=bottle
```

Or with custom parameters:

```bash
ros2 launch detection_client detection.launch.py \
  model_name:=megapose-1.0-RGB-multi-hypothesis \
  mesh_dir:=/absolute/path/to/meshes \
  mesh_units:=mm \
  yolo_model:=yolov8n.pt \
  yolo_confidence:=0.5 \
  target_label:=bottle \
  object_label:=my_bottle \
  pose_score_threshold:=0.3
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `megapose-1.0-RGB` | MegaPose model variant (see below) |
| `mesh_dir` | `<package>/meshes` | Path to directory containing object mesh folders |
| `mesh_units` | `mm` | Units of the mesh vertex coordinates (`mm` or `m`) |
| `yolo_model` | `yolov8n.pt` | YOLO model name or path (auto-downloads from Ultralytics) |
| `yolo_confidence` | `0.5` | YOLO detection confidence threshold |
| `target_label` | `bottle` | YOLO class name to detect (must be a COCO class for default model) |
| `object_label` | same as `target_label` | Label matching the mesh folder name in `mesh_dir` |
| `pose_score_threshold` | `0.3` | MegaPose score below which tracking re-initialises via YOLO |
| `rgb_topic` | `/camera/camera/color/image_raw` | RGB image topic |
| `depth_topic` | `/camera/camera/depth/image_rect_raw` | Depth image topic |
| `info_topic` | `/camera/camera/color/camera_info` | Camera info topic |

### Available MegaPose models

| Model | Depth required | Hypotheses | Notes |
|---|---|---|---|
| `megapose-1.0-RGB` | No | 1 | Fastest, RGB only |
| `megapose-1.0-RGBD` | Yes | 1 | Uses depth for refinement |
| `megapose-1.0-RGB-multi-hypothesis` | No | 5 | More accurate, slower init |
| `megapose-1.0-RGB-multi-hypothesis-icp` | Yes | 5 | Best accuracy, requires depth + ICP |

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/tf` | `tf2_msgs/TFMessage` | Transform: camera frame -> `detected_object` |
| `~/object_marker` | `visualization_msgs/Marker` | Green cube marker at the estimated pose |

## Visualisation in RViz

1. Open RViz: `rviz2`
2. Set **Fixed Frame** to your camera's optical frame (e.g. `camera_color_optical_frame`)
3. Add a **TF** display to see the `detected_object` frame
4. Add a **Marker** display subscribing to `/detection_node/object_marker`

## Package Structure

```
megaposeClient/
  package.xml                     # ROS2 package manifest
  setup.py                        # Entry points
  setup.cfg                       # Install config
  resource/detection_client       # Ament resource index marker
  detection_client/
    __init__.py
    detection_node.py             # Single node with all logic
  launch/
    detection.launch.py           # Launch file with all parameters
  meshes/                         # Place object meshes here
```

## License

Apache-2.0
