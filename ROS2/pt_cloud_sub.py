import open3d as o3d
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import time

class PointCloudProcessor(Node):
    def __init__(self):
        self.last_time = time.time()  # Initialiser la dernière fois
        super().__init__('point_cloud_processor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.listener_callback,
            10)

    def listener_callback(self, msg):

        current_time = time.time()
        elapsed_time = current_time - self.last_time
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            print("FPS:", fps)
        self.last_time = current_time

        # Convertir PointCloud2 en Open3D point cloud
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        np_points = np.array(points)
        
        """
        print(len(np_points))
        print(np_points.ndim)
        print(np_points.dtype.names)
        """
        # Convertir l'array structuré en un array 2D
        try:
            np_points = np.column_stack((np_points['x'], np_points['y'], np_points['z']))
        except ValueError:
            print("Erreur conversion 1D -> 2D")
            
        
        
        
        # Créer une carte de profondeur
        depth_map = self.create_depth_map(np_points)

        # Application de la carte de couleurs
        colored_depth_map = self.apply_colormap_to_depth_map(depth_map)

        # Détection d'objets ici à faire

        # Affichage des différentes vues
        #cv2.imshow('Nuage de Points 3D', ...)
        #cv2.imshow('Détection objets', ...)
        cv2.imshow('Carte de profondeur', depth_map)
        cv2.imshow('Carte de profondeur coloriée', colored_depth_map)
        cv2.waitKey(1)
        

    def create_depth_map(self, pcd):
        x_values = pcd[:, 0]
        y_values = pcd[:, 1]
        z_values = pcd[:, 2]

        # Normaliser les valeurs x et y
        x_values_normalized = (x_values - x_values.min()) / (x_values.max() - x_values.min())
        y_values_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

        # Convertir en indices d'image
        img_width, img_height = 848, 480
        x_indices = np.round(x_values_normalized * (img_width - 1)).astype(int)
        y_indices = np.round(y_values_normalized * (img_height - 1)).astype(int)

        # Normaliser les valeurs de profondeur
        z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min()) * 255
        z_normalized = z_normalized.astype(np.uint8)

        # Créer une image de profondeur vide
        depth_image = np.zeros((img_height, img_width), dtype=np.uint8)

        # Utiliser l'indexation avancée pour remplir l'image
        depth_image[y_indices, x_indices] = z_normalized

        return depth_image
        
    
    def apply_colormap_to_depth_map(self, depth_map):
        # Normaliser la depth map (0-255)
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = np.uint8(normalized_depth)

        # Appliquer la carte de couleurs
        colored_depth_map = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        return colored_depth_map

def main(args=None):
    rclpy.init(args=args)
    point_cloud_processor = PointCloudProcessor()
    rclpy.spin(point_cloud_processor)
    point_cloud_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
