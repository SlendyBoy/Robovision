import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import time
from cv_bridge import CvBridge, CvBridgeError
import cupy as cp

class PointCloudProcessor(Node):
    def __init__(self):

        super().__init__('pointcloud_proc') #noeud

        self.last_time_pc = time.time()  # Initialiser last FPS
        self.bridge = CvBridge()
        self.rgb_image = None

        self.depth_intrinsics = cp.array([
            [427.3677673339844, 0.0, 428.515625],
            [0.0, 427.3677673339844, 237.0117950439453],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = cp.array([-0.05677277222275734, 
                                    0.06541391462087631, 
                                    -0.0005386002594605088, 
                                    0.0006388379260897636, 
                                    -0.021308405324816704
        ])
        
        self.camera_matrix = cp.array([
            [641.3475952148438, 0.0, 651.397705078125], 
            [0.0, 639.782958984375, 359.14453125], 
            [0.0, 0.0, 1.0]
        ])

        # Extrinsèques (rotation et translation) de la caméra de profondeur vers la caméra RGB
        self.rotation_matrix = cp.array([
            [0.9999995231628418, -0.0008866226417012513, 0.0003927878278773278], 
            [0.000888532551471144, 0.9999876618385315, -0.004889312665909529], 
            [-0.00038844801019877195, 0.004889659583568573, 0.9999879598617554]
        ])
        self.translation_vector = cp.array([-0.05912087857723236, 0.0001528092980151996, 6.889161159051582e-05])

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
        
        self.depth_image_raw_pub = self.create_publisher(
            Image, 
            '/robovision/depth/raw', 
            10)
        
        self.depth_image_color_pub = self.create_publisher(
            Image, 
            '/robovision/depth/color', 
            10)
        

    def cv2_txt(self, frame, txt, x, y, size):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 1, cv2.LINE_AA)

    def listener_rgb_image(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.undistort(self.rgb_image, cp.asnumpy(self.camera_matrix), cp.asnumpy(self.dist_coeffs))


    def publish_depth_image_raw(self, depth_image):
        try:
            # Convertir l'image de profondeur raw en un message ROS
            depth_image_raw_msg = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
            self.depth_image_raw_pub.publish(depth_image_raw_msg)
        except CvBridgeError as e:
            print(e)


    def publish_depth_image_color(self, depth_image):
        try:
            # Convertir l'image de profondeur RGB en un message ROS
            depth_image_color_msg = self.bridge.cv2_to_imgmsg(depth_image, "bgr8")
            self.depth_image_color_pub.publish(depth_image_color_msg)
        except CvBridgeError as e:
            print(e)



    # pointcloud sur image RGB
    def overlay_points_on_image(self, points_2d):
        # Créer un masque vide
        mask = np.zeros(self.rgb_image.shape, dtype=np.uint8)

        # Filtrer les points qui sont à l'intérieur des limites de l'image
        valid_points = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < self.rgb_image.shape[1]) & \
                    (0 <= points_2d[:, 1]) & (points_2d[:, 1] < self.rgb_image.shape[0])

        # Compter le nombre de points valides
        num_valid_points = np.sum(valid_points)

        # Coordonnées des points valides
        valid_points_2d = points_2d[valid_points].astype(int)

        # Dessiner les points sur le masque
        mask[valid_points_2d[:, 1], valid_points_2d[:, 0]] = [0, 255, 0]

        # Fusionner le masque avec l'image RGB
        overlayed_image = cv2.addWeighted(self.rgb_image, 1, mask, 1, 0)

        cv2.imshow('RGB with Point Cloud Overlay', overlayed_image)

    
    def project_points_to_image(self, points_3d, rotation_matrix, translation_vector, camera_matrix):

        # Convertir les données structurées en tableau NumPy standard
        x = points_3d['x'].flatten()
        y = points_3d['y'].flatten()
        z = points_3d['z'].flatten()
        points_3d_np = np.column_stack((x, y, z))

        # Convertir le tableau NumPy en tableau CuPy
        points_3d_cp = cp.asarray(points_3d_np)

        # Appliquer les extrinsèques
        points_3d_transformed = cp.dot(points_3d_cp, rotation_matrix.T) + translation_vector

        # Ajouter une colonne de 1 pour les coordonnées homogènes
        ones = cp.ones((points_3d_transformed.shape[0], 1), dtype=cp.float64)
        points_3d_homogeneous = cp.hstack([points_3d_transformed, ones])

        # Effectuer la multiplication matricielle
        points_2d_homogeneous = cp.dot(camera_matrix, points_3d_homogeneous[:, :3].T).T

        # Normaliser pour obtenir les coordonnées u, v
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

        return cp.asnumpy(points_2d)  # Convertir le résultat en tableau NumPy
    
    
    # depth image
    def create_depth_image(self, points_3d, points_2d):
        depth_image_cp = cp.full(self.rgb_image.shape[:2], 10000, dtype=cp.float32)

        # Convertir les données structurées en tableau NumPy standard
        x = points_3d['x'].flatten()
        y = points_3d['y'].flatten()
        z = points_3d['z'].flatten()
        points_3d_np = np.column_stack((x, y, z))

        # Convertir en tableau CuPy
        points_3d_cp = cp.asarray(points_3d_np)
        points_2d_cp = cp.asarray(points_2d)

        # Coordonnées et valeurs de profondeur valides
        valid_points = (0 <= points_2d_cp[:, 0]) & (points_2d_cp[:, 0] < self.rgb_image.shape[1]) & \
                    (0 <= points_2d_cp[:, 1]) & (points_2d_cp[:, 1] < self.rgb_image.shape[0])
        valid_points_2d = points_2d_cp[valid_points].astype(int)
        valid_z = points_3d_cp[valid_points][:, 2]

        # Remplir l'image de profondeur
        depth_image_cp[valid_points_2d[:, 1], valid_points_2d[:, 0]] = valid_z

        return cp.asnumpy(depth_image_cp)  # Convertir en tableau NumPy pour le traitement ultérieur

    def display_depth_image(self, depth_image):
        # Exclure les valeurs sans données pour trouver la plage de normalisation
        min_depth = cp.min(depth_image[depth_image != 10000])
        max_depth = cp.max(depth_image[depth_image != 10000])

        # Inverser la normalisation pour que les objets proches soient rouges
        depth_normalized = cp.clip((max_depth - depth_image) / (max_depth - min_depth), 0, 1)
        depth_normalized = (depth_normalized * 255).astype(cp.uint8)

        # Définir les pixels sans données sur noir (valeur 0)
        depth_normalized[depth_image == 10000] = 0

        # Appliquer une colormap
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        self.publish_depth_image_color(depth_colormap) # Publier l'image de profondeur RGB

        cv2.imshow('Depth Image', depth_colormap)


    def listener_pointcloud(self, msg):

        current_time = time.time()
        elapsed_time = current_time - self.last_time_pc
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
        self.last_time_pc = current_time

        # Convertir PointCloud2 en array np
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False, reshape_organized_cloud=True)

        if self.rgb_image is not None:
            # Projeter les points sur l'image RGB
            points_2d = self.project_points_to_image(points, self.rotation_matrix, self.translation_vector, self.camera_matrix)
            #self.overlay_points_on_image(points_2d)

            depth_image = self.create_depth_image(points, points_2d)
            self.publish_depth_image_raw(depth_image) # Publier l'image de profondeur raw
            self.display_depth_image(depth_image)

        # Gérer les entrées clavier
        key = cv2.waitKey(1) & 0xFF

def main(args=None):
    rclpy.init(args=args)
    point_cloud_processor = PointCloudProcessor()
    rclpy.spin(point_cloud_processor)
    point_cloud_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
