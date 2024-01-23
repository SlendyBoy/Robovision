from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Chemin vers le dossier de partage de votre package
    pkg_share = get_package_share_directory('robovision_ros')
    urdf_file = os.path.join(pkg_share, 'urdf', 'cam.urdf.xml')
    print("URDF file path: ", urdf_file)

    return LaunchDescription([
        DeclareLaunchArgument(
            'urdf_file',
            default_value=urdf_file,
            description='Path to URDF file'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': Command(['xacro ', LaunchConfiguration('urdf_file')])}]
        ),
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
        ),
        Node(
            package='robovision_ros',
            namespace='robovision_ros',
            executable='face_reco_analysis',
            name='face_reco_analysis'
        )
    ])
