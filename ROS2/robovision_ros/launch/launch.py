from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robovision_ros',
            namespace='robovision_ros',
            executable='pointcloud_proc',
            name='pointcloud_proc'
        ),
        Node(
            package='robovision_ros',
            namespace='robovision_ros',
            executable='vision_obj_pers',
            name='vision_obj_pers'
        )
    ])
