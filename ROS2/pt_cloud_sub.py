import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import time
from cv_bridge import CvBridge

class PointCloudProcessor(Node):
    def __init__(self):

        super().__init__('point_cloud_processor') #noeud


        self.last_time_pc = time.time()  # Initialiser last FPS
        self.bridge = CvBridge()

        self.zoom_factor = 1  # niveau de zoom de base (100%)
        self.max_point_size = 2 # Attribut taille points

        #cv2.namedWindow("Carte de profondeur RGB", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow("Carte de profondeur RGB")
        cv2.setMouseCallback("Carte de profondeur RGB", self.zoom_callback)

        self.use_fixed_limits = True  # Commencer avec le mode 'avec limites'

        # les subs
        self.subscription_pt_cloud = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.listener_pointcloud,
            10)
        
        self.subscription_camera_rgb = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_cam,
            10
        )

    def zoom_callback(self, event, x, y, flags, param):

        # Zoom avant avec le clic gauche
        if event == cv2.EVENT_LBUTTONDOWN:
            self.zoom_factor = min(self.zoom_factor + 0.1, 3)

        # Zoom arrière avec le clic droit
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zoom_factor = max(self.zoom_factor - 0.1, 1)

    def apply_zoom(self, image):
        original_size = (image.shape[1], image.shape[0])  # Largeur, Hauteur
        zoomed_size = (int(original_size[0] * self.zoom_factor), int(original_size[1] * self.zoom_factor))

        # Redimensionner l'image
        zoomed_img = cv2.resize(image, zoomed_size, interpolation=cv2.INTER_LINEAR)

        # Recadrer l'image si elle est agrandie
        if self.zoom_factor > 1:
            center_x, center_y = zoomed_size[0] // 2, zoomed_size[1] // 2
            x_start = max(center_x - original_size[0] // 2, 0)
            y_start = max(center_y - original_size[1] // 2, 0)
            zoomed_img = zoomed_img[y_start:y_start + original_size[1], x_start:x_start + original_size[0]]

        return zoomed_img
        

    def listener_cam(self, msg):

        try:
            # Convertir le message ROS en une image OpenCV
            cam = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Camera Image", cam)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error('Erreur lors de la conversion de l\'image ROS en image OpenCV: %r' % (e,))

    def cv2_txt(self, frame, txt, x, y, size):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 1, cv2.LINE_AA)
        


    def listener_pointcloud(self, msg):

        current_time = time.time()
        elapsed_time = current_time - self.last_time_pc
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
        self.last_time_pc = current_time

        # Convertir PointCloud2 en array np
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False, reshape_organized_cloud=True)
        
        np_points = np.array(points)

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
        #cv2.imshow('Détection objets', ...)
        #cv2.imshow('Carte de profondeur', depth_map)

        
        # Appliquer le zoom
        zoom_color_depth = self.apply_zoom(colored_depth_map)

        # Afficher les FPS
        self.cv2_txt(zoom_color_depth, f'FPS: {round(fps, 2)}', 10, 30, 1)
        self.cv2_txt(zoom_color_depth, f'Pts: {points.size}', 220, 30, 0.4)
        self.cv2_txt(zoom_color_depth, f'Zoom: {round(self.zoom_factor * 100)}%', 10, 60, 0.4)
        self.cv2_txt(zoom_color_depth, f'Pt size: {self.max_point_size}', 110, 60, 0.4)
        self.cv2_txt(zoom_color_depth, f"Mode: {'limit' if self.use_fixed_limits else 'no limit'}", 220, 60, 0.4)

        cv2.imshow('Carte de profondeur RGB', zoom_color_depth)

        # Gérer les entrées clavier
        key = cv2.waitKey(1) & 0xFF
        if key == ord('z'):
            self.max_point_size = min(self.max_point_size + 1, 10)  # Augmenter max_point_size
        elif key == ord('s'):
            self.max_point_size = max(self.max_point_size - 1, 1)  # Diminuer max_point_size
        elif key == ord('m'):
            self.use_fixed_limits = not self.use_fixed_limits # modif mode limite
        

    def create_depth_map(self, pcd):

        x_values = pcd[:, 0]
        y_values = pcd[:, 1]
        z_values = pcd[:, 2]

        if self.use_fixed_limits:

            # AVEC LIMITE
            x_min, x_max = -5, 5
            y_min, y_max = -5, 5  
            z_min, z_max = -10, 10  

            # Normalisation des axes X et Y
            x_values_normalized = ((x_values - x_min) / (x_max - x_min))
            y_values_normalized = ((y_values - y_min) / (y_max - y_min))

            # Normalisation inversée pour Z
            z_normalized = (z_max - z_values) / (z_max - z_min) * 255

            # Clip
            x_values_normalized = np.clip(x_values_normalized, 0, 1)
            y_values_normalized = np.clip(y_values_normalized, 0, 1)
            z_normalized = np.clip(z_normalized, 0, 255)

            # Convertir en indices d'image et créer l'image de profondeur
            img_width, img_height = 1280, 720
            x_indices = np.round(x_values_normalized * (img_width - 1)).astype(int)
            y_indices = np.round(y_values_normalized * (img_height - 1)).astype(int)
            z_normalized = z_normalized.astype(np.uint8)

            # Init de la carte vide et remplissage
            depth_image = np.zeros((img_height, img_width), dtype=np.uint8)

        else:
            # SANS LIMITES

            """print(f"X min: {x_values.min()}")
            print(f"X max: {x_values.max()}")

            print(f"Y min: {y_values.min()}")
            print(f"Y max: {y_values.max()}")

            print(f"Z min: {z_values.min()}")
            print(f"Z max: {z_values.max()}")"""

            # Normaliser les valeurs x et y
            x_values_normalized = (x_values - x_values.min()) / (x_values.max() - x_values.min())
            y_values_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

            # Convertir en indices d'image
            img_width, img_height = 1280, 720
            x_indices = np.round(x_values_normalized * (img_width - 1)).astype(int)
            y_indices = np.round(y_values_normalized * (img_height - 1)).astype(int)

            # Normaliser les valeurs de profondeur
            z_normalized = 1 - (z_values - z_values.min()) / (z_values.max() - z_values.min()) * 255
            z_normalized = z_normalized.astype(np.uint8)

            # Créer une image de profondeur vide
            depth_image = np.zeros((img_height, img_width), dtype=np.uint8)


        # Calculer la taille des points en fonction de leur profondeur normalisée
        # Les points plus proches (z_normalized petit) auront une taille plus grande
        point_sizes = np.round((1 - z_normalized / 255) * self.max_point_size)

        # Appliquer le kernel à chaque point
        # Itérer sur chaque taille unique présente dans point_sizes
        for size in np.unique(point_sizes):

            # Kernel carré de la taille du point, utilisé pour dilater les points
            kernel = np.ones((int(size), int(size)), np.uint8)

            # Image temporaire
            points_to_dilate = np.zeros_like(depth_image)

            # Assigner les valeurs normalisées de Z aux points qui correspondent à la taille de kernel actuelle
            points_to_dilate[y_indices[point_sizes == size], x_indices[point_sizes == size]] = z_normalized[point_sizes == size]

            # Agrandir les points en utilisant le kernel
            dilated_points = cv2.dilate(points_to_dilate, kernel, iterations=1)

            # MAJ l'image de profondeur originale avec les points dilatés
            # 'np.maximum' garantit que la valeur la plus élevée entre l'image de profondeur actuelle et les points dilatés est conservée
            depth_image = np.maximum(depth_image, dilated_points)
        
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
