# Projet robotique

## Sujet 3 - Brique vision

Tanguy FROUIN 5IRC

1. État de l'Art

Sources:  

https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-using-ros-2-and-tao-pointpillars/  
https://www.stereolabs.com/docs/ros2/depth-sensing/  
https://github.com/IntelRealSense/librealsense/tree/development/wrappers/python/examples  
http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html  
https://academy.visualcomponents.com/lessons/import-point-cloud-with-python-api/  
https://www.youtube.com/watch?v=HIRj5pH2t-Y  

https://github.com/google/mediapipe/blob/master/docs/solutions/  
https://github.com/googlesamples/mediapipe/tree/main/examples  
https://developers.google.com/mediapipe/solutions/guide  

https://github.com/ultralytics/ultralytics/issues/2028  
  
https://jeffzzq.medium.com/ros2-image-pipeline-tutorial-3b18903e7329  
https://medium.com/@regis.loeb/playing-with-point-clouds-for-3d-object-detection-eff1d98e526a  
https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f  
  
http://www.diva-portal.org/smash/get/diva2:1245296/FULLTEXT01.pdf  
http://www.open3d.org/docs/release/tutorial/geometry/rgbd_image.html  
https://github.com/jamohile/stereoscopic-point-clouds  
https://github.com/strawlab/python-pcl  
https://towardsdatascience.com/how-to-use-pointnet-for-3d-computer-vision-in-an-industrial-context-3568ba37327e  

2. Développement de l'API de Vision 3D

    - Conception de l'API : API générique qui peut traiter les données de différentes caméras 3D. Cette API devra être capable d'interfacer avec ROS2 et de traiter les données de point cloud.
    - Abstraction de la caméra : traiter les données de la caméra de manière générique, permettant de se connecter à différentes caméras 3D.

3. Traitement des Données de Point Cloud

    - Prétraitement
    - Extraction des caractéristiques : Deep learning pour identifier et caractériser les personnes (TF, attributs, posture) et les objets (TF, classification taxonomique).

4. Reconnaissance et caractérisation

    - Détection de personnes et d'objets : CNN adaptés aux données 3D pour la détection et la classification.
    - Estimation des attributs et postures : Pareil pour estimer les attributs des personnes et leur posture.

5. Intégration avec ROS2

    - Noeuds ROS2 : intègrent l'API de vision 3D et fait le lien avec les fonctions de ROS2
    - Tests et calibration : Test avec la caméra 3D couleurs Intel RealSense D455 et voire Kinect

6. Documentation et Tests

    - Documentation
    - Tests avec différentes caméras pour valider l'approche générique

7. Technologies et outils potentiels

    - ROS2 : Intégration et communication
    - Python : API et algorithmes
    - TensorFlow ou PyTorch : Deep Learning, modèles traitant les données 3D
    - PCL (Point Cloud Library) : Traitement des données de points cloud
    - OpenCV : Traitement d'image et la vision par ordinateur
    - YOLOv8 : Reconnaissance d'objets, personnes
    - Mediapipe : Pareil que YOLOv8 mais la solution de pose/posture n'est limitée qu'à une seule personne trackée