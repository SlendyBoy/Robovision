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

        super().__init__('pointcloud_proc') #noeud


        self.last_time_pc = time.time()  # Initialiser last FPS
        self.bridge = CvBridge()
        self.rgb_image = None

        self.zoom_factor = 1  # niveau de zoom de base (100%)
        self.max_point_size = 2 # Attribut taille points PC
        self.use_fixed_limits = True  # Commencer avec le mode 'avec limites'

        self.camera_intrinsics = np.array([
            [641.3475952148438, 0.0, 651.397705078125],
            [0.0, 639.782958984375, 359.14453125],
            [0.0, 0.0, 1.0]
        ])
        self.depth_intrinsics = np.array([
            [427.3677673339844, 0.0, 428.515625],
            [0.0, 427.3677673339844, 237.0117950439453],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([-0.05677277222275734, 
                                    0.06541391462087631, 
                                    -0.0005386002594605088, 
                                    0.0006388379260897636, 
                                    -0.021308405324816704])
        
        self.camera_matrix = np.array([
            [641.3475952148438, 0.0, 651.397705078125], 
            [0.0, 639.782958984375, 359.14453125], 
            [0.0, 0.0, 1.0]
        ])

        # Extrinsèques (rotation et translation) de la caméra de profondeur vers la caméra RGB
        self.rotation_matrix = np.array([
            [0.9999995231628418, -0.0008866226417012513, 0.0003927878278773278], 
            [0.000888532551471144, 0.9999876618385315, -0.004889312665909529], 
            [-0.00038844801019877195, 0.004889659583568573, 0.9999879598617554]
        ])
        self.translation_vector = np.array([-0.05912087857723236, 0.0001528092980151996, 6.889161159051582e-05])

        cv2.namedWindow("Carte de profondeur RGB", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback("Carte de profondeur RGB", self.zoom_callback)
        

        # subs PC
        self.subscription_pt_cloud = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.listener_pointcloud,
            10)
        
        self.subscription_rgb = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_rgb_image,
            10)
        

    def cv2_txt(self, frame, txt, x, y, size):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 1, cv2.LINE_AA)

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
        

    def listener_rgb_image(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.undistort(self.rgb_image, self.camera_matrix, self.dist_coeffs)



    def overlay_points_on_image(self, points_2d):
        # Créer un masque vide
        mask = np.zeros(self.rgb_image.shape, dtype=np.uint8)

        # Filtrer les points qui sont à l'intérieur des limites de l'image
        valid_points = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < self.rgb_image.shape[1]) & \
                    (0 <= points_2d[:, 1]) & (points_2d[:, 1] < self.rgb_image.shape[0])

        # Compter le nombre de points valides
        num_valid_points = np.sum(valid_points)
        print(f"Nombre de points valides affichés : {num_valid_points}")

        # Coordonnées des points valides
        valid_points_2d = points_2d[valid_points].astype(int)

        # Dessiner les points sur le masque
        mask[valid_points_2d[:, 1], valid_points_2d[:, 0]] = [0, 255, 0]

        # Fusionner le masque avec l'image RGB
        overlayed_image = cv2.addWeighted(self.rgb_image, 1, mask, 1, 0)

        cv2.imshow('RGB with Point Cloud Overlay', overlayed_image)

    def create_depth_image(self, points_3d, points_2d):
        depth_image = np.full(self.rgb_image.shape[:2], np.inf, dtype=np.float32)

        # Assurer que les points sont dans les limites de l'image
        valid_points = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < self.rgb_image.shape[1]) & \
                    (0 <= points_2d[:, 1]) & (points_2d[:, 1] < self.rgb_image.shape[0])

        # Coordonnées et valeurs de profondeur valides
        valid_points_2d = points_2d[valid_points].astype(int)
        valid_z = points_3d[valid_points][:, 2]  # Extraire la composante z

        # Remplir l'image de profondeur
        depth_image[valid_points_2d[:, 1], valid_points_2d[:, 0]] = valid_z

        return depth_image


    def project_points_to_image(self, points_3d):
        # Extraire les champs x, y, z et les aplatir
        x = points_3d['x'].flatten()
        y = points_3d['y'].flatten()
        z = points_3d['z'].flatten()

        # tableau 2D standard
        points_3d = np.column_stack((x, y, z))

        # Appliquer les extrinsèques
        points_3d_transformed = np.dot(points_3d, self.rotation_matrix.T) + self.translation_vector

        # Ajouter une colonne de 1 pour les coordonnées homogènes
        ones = np.ones((points_3d_transformed.shape[0], 1), dtype=np.float64)
        points_3d_homogeneous = np.hstack([points_3d_transformed, ones])

        # Effectuer la multiplication matricielle
        # Retirer la quatrième dimension (les ones cf ci-dessus) avant de multiplier
        points_2d_homogeneous = np.dot(self.camera_intrinsics, points_3d_homogeneous[:, :3].T).T

        # Normaliser pour obtenir les coordonnées u, v
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

        return points_2d

    def listener_pointcloud(self, msg):

        current_time = time.time()
        elapsed_time = current_time - self.last_time_pc
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
        self.last_time_pc = current_time

        # Convertir PointCloud2 en array np
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False, reshape_organized_cloud=True)

        if self.rgb_image is not None:
            # Projetons les points sur l'image RGB
            points_2d = self.project_points_to_image(points)
            self.overlay_points_on_image(points_2d)

        """# Créer une carte de profondeur
        depth_map = self.create_depth_map(points)

        # Application de la carte de couleurs
        colored_depth_map = self.apply_colormap_to_depth_map(depth_map)
 
        # Appliquer le zoom
        zoom_color_depth = self.apply_zoom(colored_depth_map)

        # Afficher les FPS
        self.cv2_txt(zoom_color_depth, f'FPS: {round(fps, 2)}', 10, 30, 1)
        self.cv2_txt(zoom_color_depth, f'Pts: {points["x"].size}', 220, 30, 0.4)
        self.cv2_txt(zoom_color_depth, f'Zoom: {round(self.zoom_factor * 100)}%', 10, 60, 0.4)
        self.cv2_txt(zoom_color_depth, f'Pt size: {self.max_point_size}', 110, 60, 0.4)
        self.cv2_txt(zoom_color_depth, f"Mode: {'limit' if self.use_fixed_limits else 'no limit'}", 220, 60, 0.4)

        cv2.imshow('Carte de profondeur RGB', zoom_color_depth)"""

        # Gérer les entrées clavier
        key = cv2.waitKey(1) & 0xFF
        if key == ord('z'):
            self.max_point_size = min(self.max_point_size + 1, 10)  # Augmenter max_point_size
        elif key == ord('s'):
            self.max_point_size = max(self.max_point_size - 1, 1)  # Diminuer max_point_size
        elif key == ord('m'):
            self.use_fixed_limits = not self.use_fixed_limits # modif mode limite
        

    def create_depth_map(self, pcd):

        x_values = np.array(pcd['x'], dtype=np.float32)
        y_values = np.array(pcd['y'], dtype=np.float32)
        z_values = np.array(pcd['z'], dtype=np.float32)

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
