# MegaPose Client

ROS2 node for real-time 6-DOF object pose estimation using [MegaPose6D](https://github.com/megapose6d/megapose6d) with YOLO-based object detection. Subscribes to a RealSense D455 camera, detects objects, estimates their 6D pose, and publishes results as TF transforms, RViz markers, and a contour overlay image.

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
              ┌─────────┼─────────┐
              ▼         ▼         ▼
         TF Broadcast  Marker  Contour Overlay
```

The node operates in two phases:

1. **Initialisation** -- YOLO detects the target object and produces a bounding box. MegaPose runs the full pipeline (coarse SO(3) grid search + refiner). This is slow (~1-2s) but only happens once.
2. **Tracking** -- Subsequent frames skip detection and coarse estimation entirely. The previous pose is passed directly to MegaPose's refiner for fast iterative updates. If the pose confidence drops below a threshold, the node automatically re-initialises via YOLO.

## Prerequisites

- **ROS2** (Humble or later)
- **MegaPose6D** installed as a Python package:
  ```bash
  cd /path/to/megapose6d
  pip install -e .
  ```
- **MegaPose model weights** downloaded to `megapose6d/local_data/megapose-models/`:
  ```bash
  python -m megapose.scripts.download --megapose_models
  ```
- **Ultralytics** (YOLO) and **trimesh** (overlay rendering):
  ```bash
  pip install ultralytics trimesh
  ```
- **RealSense ROS2 driver**:
  ```bash
  sudo apt install ros-${ROS_DISTRO}-realsense2-camera
  ```
- **NVIDIA GPU** with CUDA (required by MegaPose and YOLO)

## Quick Start

### 1. Prepare your object mesh

Place a 3D mesh (`.ply` or `.obj`) in the `meshes/` directory. The folder name becomes the object label automatically:

```
meshes/
  └── my_object/
        └── model.ply
```

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

Simplest form -- just drop your mesh and go, YOLO grabs whatever it sees:

```bash
ros2 launch detection_client detection.launch.py
```

Filter to a specific YOLO class if there are multiple objects in the scene:

```bash
ros2 launch detection_client detection.launch.py target_label:=bottle
```

Full example with tuning for speed:

```bash
ros2 launch detection_client detection.launch.py \
  model_name:=megapose-1.0-RGB \
  n_refiner_iterations:=2 \
  n_pose_hypotheses:=1 \
  mesh_dir:=/absolute/path/to/meshes
```

## Parameters

### General

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `megapose-1.0-RGB` | MegaPose model variant (see table below) |
| `mesh_dir` | `<package>/meshes` | Path to directory containing object mesh folders |
| `mesh_units` | `mm` | Units of the mesh vertex coordinates (`mm` or `m`) |
| `object_label` | auto-detected | Mesh subfolder name. If empty, picks the first subfolder in `mesh_dir` |

### Detection

| Parameter | Default | Description |
|---|---|---|
| `yolo_model` | `yolov8n.pt` | YOLO model name or path (auto-downloads from Ultralytics) |
| `yolo_confidence` | `0.5` | YOLO detection confidence threshold |
| `target_label` | `""` (any) | YOLO class name to filter for. Empty = use highest-confidence detection regardless of class |
| `pose_score_threshold` | `0.3` | MegaPose score below which tracking re-initialises via YOLO |

### MegaPose Tuning

| Parameter | Default | Description |
|---|---|---|
| `n_refiner_iterations` | `-1` (model default) | Number of refiner iterations. Fewer = faster. The main FPS lever |
| `n_pose_hypotheses` | `-1` (model default) | Coarse hypotheses to refine. More = better init, slower |
| `bsz_images` | `128` | Coarse model batch size (GPU memory tradeoff) |
| `bsz_objects` | `8` | Refiner batch size (GPU memory tradeoff) |

### Topics & Overlay

| Parameter | Default | Description |
|---|---|---|
| `rgb_topic` | `/camera/camera/color/image_raw` | RGB image topic |
| `depth_topic` | `/camera/camera/depth/image_rect_raw` | Depth image topic |
| `info_topic` | `/camera/camera/color/camera_info` | Camera info topic |
| `publish_overlay` | `true` | Publish mesh contour overlay on `~/overlay` |

### Available MegaPose Models

| Model | Depth | Hypotheses | Notes |
|---|---|---|---|
| `megapose-1.0-RGB` | No | 1 | Fastest, RGB only |
| `megapose-1.0-RGBD` | Yes | 1 | Uses depth for refinement |
| `megapose-1.0-RGB-multi-hypothesis` | No | 5 | More accurate, slower init |
| `megapose-1.0-RGB-multi-hypothesis-icp` | Yes | 5 | Best accuracy, requires depth + ICP |

## Expected FPS (Tracking Phase)

| `n_refiner_iterations` | Approx. time per frame | FPS |
|---|---|---|
| 5 (default) | 200 - 350ms | 3 - 5 |
| 3 | 120 - 200ms | 5 - 8 |
| 2 | 80 - 150ms | 7 - 12 |
| 1 | 50 - 100ms | 10 - 20 |

Benchmarked on RTX 3060-class GPUs. The init frame (YOLO + coarse + refiner) takes 1.5-3s regardless. The overlay adds ~2-5ms (negligible).

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/tf` | `tf2_msgs/TFMessage` | Transform: camera optical frame -> `detected_object` |
| `~/object_marker` | `visualization_msgs/Marker` | Green cube marker at the estimated pose |
| `~/overlay` | `sensor_msgs/Image` | RGB image with green mesh contour overlay |

## Visualisation in RViz

1. Open RViz: `rviz2`
2. Set **Fixed Frame** to your camera's optical frame (e.g. `camera_color_optical_frame`)
3. Add a **TF** display to see the `detected_object` frame
4. Add a **Marker** display subscribing to `/detection_node/object_marker`
5. Add an **Image** display subscribing to `/detection_node/overlay` to see the mesh contour on the camera feed

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
