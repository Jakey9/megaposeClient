from setuptools import find_packages, setup

package_name = "detection_client"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/detection.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jake.tan",
    maintainer_email="jake.tan@todo.com",
    description="YOLO detection + MegaPose6D 6-DOF pose estimation ROS2 client",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "detection_node = detection_client.detection_node:main",
        ],
    },
)
