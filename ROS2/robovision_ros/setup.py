from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robovision_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),  # Inclure les fichiers de lancement
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tanguy Frouin',
    maintainer_email='tanguy.frouin@cpe.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_proc = robovision_ros.pointcloud_proc:main',
            'vision_obj_pers = robovision_ros.vision_obj_pers:main',
            'face_reco_analysis = robovision_ros.face_reco_analysis:main'
        ],
    },
)
