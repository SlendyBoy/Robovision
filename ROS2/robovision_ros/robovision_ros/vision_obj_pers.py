import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO


# Load the YOLOv8 model
model_obj = YOLO('yolov8m.pt')
model_pose = YOLO('yolov8m-pose.pt')
#model.fuse()

class VisionObjPers(Node):
    def __init__(self):

        super().__init__('vision_obj_pers') #noeud

        self.last_time_obj = time.time()  # Initialiser last FPS
        self.bridge = CvBridge()

        self.zoom_factor = 1  # niveau de zoom de base (100%)
        self.last_zone_depth = 10000
        self.depth_image = None

        self.depth_intrinsics = np.array([
            [427.3677673339844, 0.0, 428.515625],
            [0.0, 427.3677673339844, 237.0117950439453],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([-0.05677277222275734, 
                                    0.06541391462087631, 
                                    -0.0005386002594605088, 
                                    0.0006388379260897636, 
                                    -0.021308405324816704
        ])
        
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

        cv2.namedWindow("Detection objets et posture", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        
        self.camera_rgb_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_cam,
            10
        )

        self.depth_image_raw_sub = self.create_subscription(
            Image,
            '/robovision/depth/raw',
            self.depth_image_raw_callback,
            10
        )

    def cv2_txt(self, frame, txt, x, y, size):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 2, cv2.LINE_AA)

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
        


    def project_points_to_image(self, points_3d):
        x = points_3d[:, 0].flatten()
        y = points_3d[:, 1].flatten()
        z = points_3d[:, 2].flatten()

        # tableau 2D standard
        points_3d = np.column_stack((x, y, z))

        # Appliquer les extrinsèques
        points_3d_transformed = np.dot(points_3d, self.rotation_matrix.T) + self.translation_vector
        self.get_logger().info(f"points_3d_transformed {points_3d_transformed}")

        # Ajouter une colonne de 1 pour les coordonnées homogènes
        ones = np.ones((points_3d_transformed.shape[0], 1), dtype=np.float64)
        points_3d_homogeneous = np.hstack([points_3d_transformed, ones])
        self.get_logger().info(f"points_3d_homogeneous {points_3d_homogeneous}")

        # Effectuer la multiplication matricielle
        # Retirer la quatrième dimension (les ones cf ci-dessus) avant de multiplier
        points_2d_homogeneous = np.dot(self.camera_matrix, points_3d_homogeneous[:, :3].T).T
        self.get_logger().info(f"points_2d_homogeneous {points_2d_homogeneous}")

        # Normaliser pour obtenir les coordonnées u, v
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
        self.get_logger().info(f"points_2d normalized {points_2d}")

        return points_2d
        

    
    def draw_3d_box(self, image, bbox_2d, depth):
        
        # bbox_2d: (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = bbox_2d

        # Définir les 8 sommets du "cube" en 3D
        # même profondeur pour tous les points
        depth_3d = depth + 50  # offset pour donner une épaisseur à la boîte
        points_3d = np.array([
            [x_min, y_min, depth],
            [x_max, y_min, depth],
            [x_max, y_max, depth],
            [x_min, y_max, depth],
            [x_min, y_min, depth_3d],
            [x_max, y_min, depth_3d],
            [x_max, y_max, depth_3d],
            [x_min, y_max, depth_3d]
        ])
        # Projeter les points 3D sur l'espace 2D
        points_2d = self.project_points_to_image(points_3d)

        if points_2d.shape[0] == 8:
            points_2d = points_2d.reshape(-1, 2)

            # Filtrer les points qui sont à l'intérieur des limites de l'image
            valid_points = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < image.shape[1]) & \
                        (0 <= points_2d[:, 1]) & (points_2d[:, 1] < image.shape[0])
            
            num_valid_points = np.sum(valid_points)

            self.get_logger().info(f"num valid points {num_valid_points}")

            # Ne dessiner que les lignes pour les points valides
            for i in range(4):
                if valid_points[i] and valid_points[(i+1)%4]:
                    pt1 = (int(points_2d[i][0]), int(points_2d[i][1]))
                    pt2 = (int(points_2d[(i+1)%4][0]), int(points_2d[(i+1)%4][1]))
                    cv2.line(image, pt1, pt2, (0, 0, 255), 2)
                    self.get_logger().info(f"pt1 {pt1}")
                    self.get_logger().info(f"pt2 {pt2}")

                if valid_points[i+4] and valid_points[(i+1)%4 + 4]:
                    pt3 = (int(points_2d[i+4][0]), int(points_2d[i+4][1]))
                    pt4 = (int(points_2d[(i+1)%4 + 4][0]), int(points_2d[(i+1)%4 + 4][1]))
                    cv2.line(image, pt3, pt4, (0, 0, 255), 2)
                    self.get_logger().info(f"pt3 {pt3}")
                    self.get_logger().info(f"pt4 {pt4}")

                if valid_points[i] and valid_points[i+4]:
                    cv2.line(image, pt1, pt3, (0, 0, 255), 2)

                
                
                
        else:
            self.get_logger().info("Erreur dans la projection des points 3D en 2D")


    def get_average_depth_in_zone(self, depth_image, x_min, x_max, y_min, y_max):

        # Extraire la zone spécifiée de l'image de profondeur
        depth_zone = depth_image[y_min:y_max, x_min:x_max]

        # Créer un masque pour exclure les valeurs de 10000 (absence de données)
        valid_depth_mask = depth_zone != 10000

        # Calculer la moyenne en excluant les valeurs de 10000
        if np.any(valid_depth_mask):
            self.last_zone_depth = np.mean(depth_zone[valid_depth_mask])
        else:
            self.last_zone_depth = self.last_zone_depth  # Aucune valeur valide dans la zone

        return self.last_zone_depth

    def depth_image_raw_callback(self, msg):

        try:
            # Convertir le message ROS depth raw en image OpenCV
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Erreur de conversion CvBridge depth image: {e}')



    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)


    def listener_cam(self, msg):

        try:

            current_time = time.time()
            elapsed_time = current_time - self.last_time_obj
            if elapsed_time > 0:
                fps = 1.0 / elapsed_time
            self.last_time_obj = current_time

            # Convertir le message ROS en une image OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Appliquer la correction de distorsion sur l'image
            undistorted_frame = self.undistort_image(frame)

            # YOLOv8 obj inference
            results = model_obj(undistorted_frame, conf=0.5)

            results_np = results[0].cpu().numpy()

            datas_frame = results_np.plot()

            for result in results_np:
                boxes = result.boxes
                cls = boxes.cls

                # Coordonnées de la bounding box (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = boxes.xyxy[0]
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # Calculer la distance moyenne dans la bounding box
                average_depth = self.get_average_depth_in_zone(self.depth_image, x_min, x_max, y_min, y_max)

                # Afficher la distance au-dessus de la bounding box
                label = f"Dist: {round(float(average_depth), 2)} m"
                self.cv2_txt(datas_frame, label, x_min, y_min - 28, 1)

                self.draw_3d_box(datas_frame, (x_min, y_min, x_max, y_max), average_depth)


            #cv2.imshow("Detection objets", datas_frame)

            """# YOLOv8 pose inference
            results_pose = model_pose(datas_frame, conf=0.5)

            pose_frame = results_pose[0].plot()

            # Display the annotated frame
            self.cv2_txt(pose_frame, f'FPS: {round(fps, 2)}', 10, 30, 1)"""

            cv2.imshow("Detection objets et posture", datas_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error('Erreur lors de la conversion de l\'image ROS camera RGB en image OpenCV: %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    vision_obj_pers = VisionObjPers()
    rclpy.spin(vision_obj_pers)
    vision_obj_pers.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
