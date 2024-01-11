import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import math
import cupy as cp

# Load the YOLOv8 model
model_pose = YOLO('yolov8m-pose.pt')
model_seg = YOLO('yolov8m-seg.pt')

class VisionObjPers(Node):
    def __init__(self):

        super().__init__('vision_obj_pers') #noeud

        self.last_time_obj = time.time()  # Initialiser last FPS
        self.bridge = CvBridge()

        self.zoom_factor = 1  # niveau de zoom de base (100%)
        self.depth_image = None
        self.NOSE =         0
        self.L_EYE =        1
        self.R_EYE =        2
        self.L_EAR =        3
        self.R_EAR =        4
        self.L_SHOULDER =   5
        self.R_SHOULDER =   6
        self.L_ELBOW =      7
        self.R_ELBOW =      8
        self.LLL_WRIST =    9
        self.R_WRIST =      10
        self.L_HIP =        11
        self.R_HIP =        12
        self.L_KNEE =       13
        self.R_KNEE =       14
        self.L_ANKLE =      15
        self.R_ANKLE =      16

        self.pose_palette = np.array(
            [
                (255, 128, 0),
                (255, 153, 51),
                (255, 178, 102),
                (230, 230, 0),
                (255, 153, 255),
                (153, 204, 255),
                (255, 102, 255),
                (255, 51, 255),
                (102, 178, 255),
                (51, 153, 255),
                (255, 153, 153),
                (255, 102, 102),
                (255, 51, 51),
                (153, 255, 153),
                (102, 255, 102),
                (51, 255, 51),
                (0, 255, 0),
                (0, 0, 255),
                (255, 0, 0),
                (255, 255, 255),
            ],
            dtype=np.uint8,
        )

        self.skeleton = np.array([
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ],
            dtype=np.uint8,
        )

        self.limb_color = np.array([9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16],
            dtype=np.uint8,
        )
        self.kpt_color = np.array([16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9],
            dtype=np.uint8,
        )

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

        #cv2.namedWindow("Detection objets et posture", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        
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


    def depth_image_raw_callback(self, msg):

        try:
            # Convertir le message ROS depth raw en image OpenCV
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Erreur de conversion CvBridge depth image: {e}')


    def undistort_image(self, image):
        return cv2.undistort(image, cp.asnumpy(self.camera_matrix), cp.asnumpy(self.dist_coeffs))
    

    def calculate_object_depth(self, depth_roi, object_mask):
        object_depth = depth_roi[object_mask]
        valid_depths = object_depth[object_depth != 10000]  # Exclure les valeurs sans données
        if valid_depths.size > 0:
            return cp.mean(valid_depths)
        else:
            return None
        
    
    def calculate_and_display_object_depth(self, result, frame):
        masks = result.masks.data.cpu().numpy()  # Masques (N, H, W)
        
        if len(masks) > 0:
            mask = masks[0]
            
            # Redimensionner le masque pour correspondre à l'image de profondeur
            resized_mask = cv2.resize(mask, (self.depth_image.shape[1], self.depth_image.shape[0]))

            # Créer un masque booléen pour l'image de profondeur
            depth_mask = (resized_mask == 1)

            # Extraire les valeurs de profondeur correspondantes
            depth_values = self.depth_image[depth_mask]

            # Calculer la profondeur moyenne ou médiane pour éviter les valeurs aberrantes
            valid_depth_values = depth_values[depth_values != 10000]  # Exclure les valeurs manquantes
            if valid_depth_values.size > 0:
                object_depth = cp.median(cp.asarray(valid_depth_values))
            else:
                object_depth = cp.nan  # Aucune valeur valide trouvée

            boxes = result.boxes.cpu().numpy()
            x_min, y_min, x_max, y_max = map(int, boxes.xyxy[0])

            label = f"{round(float(object_depth), 2)} m" if not cp.isnan(object_depth) else "N/A m"
            self.cv2_txt(frame, label, x_min, y_min - 28, 1)
        else:
            # Aucun masque disponible
            self.get_logger().info("Aucun masque trouvé pour calculer la profondeur.")




    def calculate_angle(self, p1, p2, p3):
        # Calculer l'angle entre trois points
        a = cp.array(p1) - cp.array(p2)
        b = cp.array(p3) - cp.array(p2)
        angle = cp.arctan2(a[1], a[0]) - cp.arctan2(b[1], b[0])

        # Convertir en degrés et s'assurer que l'angle est entre 0 et 180 degrés
        angle = cp.degrees(angle)
        angle = abs(angle)  # prendre la valeur absolue pour éviter les angles négatifs
        if angle > 180:
            angle = 360 - angle  # ajuster si l'angle est supérieur à 180 degrés

        return angle


    def draw_angle(self, frame, p1, p2, p3, angle):
        # S'assurer que les points sont des entiers
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        p3 = (int(p3[0]), int(p3[1]))

        # Calculer les angles de début et de fin pour l'ellipse
        angle_start = math.atan2(p1[1] - p2[1], p1[0] - p2[0]) * 180 / math.pi
        angle_end = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) * 180 / math.pi

        # Ajuster les angles pour s'assurer que l'arc est le plus petit possible
        if angle_end < angle_start:
            angle_start, angle_end = angle_end, angle_start  # Intervertir si nécessaire

        # Si l'angle couvre plus de 180 degrés, dessiner l'arc plus petit
        if angle_end - angle_start > 180:
            angle_start, angle_end = angle_end, angle_start + 360

        # Dessiner l'arc de l'angle
        radius = 30
        cv2.ellipse(frame, p2, (radius, radius), 0, angle_start, angle_end, (255, 0, 0), 2)

        # Afficher le degré de l'angle
        cv2.putText(frame, str(int(angle)), (p2[0] + 10, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    
    def is_aligned_vertically(self, pt1, pt2, pt3, threshold=0.1):
        """
        Vérifie si trois points sont alignés verticalement.
        """
        # Calculer la pente entre chaque paire de points
        slope1 = abs((pt2[1] - pt1[1]) / (pt2[0] - pt1[0] + 1e-5))
        slope2 = abs((pt3[1] - pt2[1]) / (pt3[0] - pt2[0] + 1e-5))

        # Vérifier si les pentes sont proches de 0 (verticales)
        return slope1 < threshold and slope2 < threshold

    def is_aligned_horizontally(self, pt1, pt2, pt3, threshold=0.1):
        """
        Vérifie si trois points sont alignés horizontalement.
        """
        # Calculer la différence de hauteur entre chaque paire de points
        diff1 = abs(pt2[1] - pt1[1])
        diff2 = abs(pt3[1] - pt2[1])

        # Vérifier si les différences sont sous un certain seuil
        return diff1 < threshold and diff2 < threshold

    def angles_near_upright(self, angle1, angle2, threshold=20):
        """
        Vérifie si les angles sont proches de 180 degrés (debout).
        """
        return abs(angle1 - 180) < threshold and abs(angle2 - 180) < threshold

    def angles_near_sitting(self, hip_angle, knee_angle, hip_threshold=100, knee_threshold=100):
        """
        Vérifie si les angles aux hanches et aux genoux sont proches de 90 degrés (assis).
        """
        return abs(hip_angle - 90) < hip_threshold and abs(knee_angle - 90) < knee_threshold

   

    def calculate_position(self, frame, kpts):
        # Utiliser les indices pour les épaules, hanches, genoux et chevilles
        L_shoulder = kpts[self.L_SHOULDER]
        R_shoulder = kpts[self.R_SHOULDER]
        L_hip = kpts[self.L_HIP]
        R_hip = kpts[self.R_HIP]
        L_knee = kpts[self.L_KNEE]
        R_knee = kpts[self.R_KNEE]
        L_ankle = kpts[self.L_ANKLE]
        R_ankle = kpts[self.R_ANKLE]

        # Calculer les angles
        L_side_angle = self.calculate_angle(L_shoulder, L_hip, L_ankle)
        self.draw_angle(frame, L_shoulder, L_hip, L_ankle, L_side_angle)

        R_side_angle = self.calculate_angle(R_shoulder, R_hip, R_ankle)
        self.draw_angle(frame, R_shoulder, R_hip, R_ankle, R_side_angle)

        L_knee_angle = self.calculate_angle(L_hip, L_knee, L_ankle)
        self.draw_angle(frame, L_hip, L_knee, L_ankle, L_knee_angle)

        R_knee_angle = self.calculate_angle(R_hip, R_knee, R_ankle)
        self.draw_angle(frame, R_hip, R_knee, R_ankle, R_knee_angle)
        

        # Vérifier si la personne est debout
        is_standing = (self.is_aligned_vertically(L_shoulder, L_hip, L_ankle) and 
                       self.is_aligned_vertically(R_shoulder, R_hip, R_ankle)) or \
                        (L_knee_angle > 160 and R_knee_angle > 160)

        # Vérifier si la personne est assise
        is_sitting = self.angles_near_sitting(L_knee_angle, R_knee_angle) and \
                    not self.is_aligned_vertically(L_shoulder, L_hip, L_ankle) and \
                    not self.is_aligned_vertically(R_shoulder, R_hip, R_ankle)

        # Vérifier si la personne est couchée
        is_lying_down = L_side_angle < 45 or R_side_angle < 45

        # Déterminer la position
        if is_standing:
            position = "Standing"
        elif is_sitting:
            position = "Sitting"
        elif is_lying_down:
            position = "Lying down"
        else:
            position = "N/A pos"

        return [position, int(L_side_angle), int(R_side_angle), int(L_knee_angle), int(R_knee_angle)]


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

            # Detection et segmentation
            results_seg = model_seg(undistorted_frame, conf=0.4, verbose=False, retina_masks=False)

            seg_frame = results_seg[0].plot(boxes=True, masks=False)

            for result in results_seg[0]:
                self.calculate_and_display_object_depth(result, seg_frame)


            results_pose = model_pose(undistorted_frame, conf=0.5, verbose=False)
            #pose_frame = results_pose[0].plot(boxes=False)

            # Create a black frame
            black_frame = np.zeros_like(undistorted_frame)

            for result in results_pose[0]:
                
                # Det position pers
                kpts = result.keypoints.xy.cpu().numpy()
                pos = self.calculate_position(black_frame, kpts[0])
                
                # Afficher pos
                boxes = result.boxes.cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, boxes.xyxy[0])
                
                cv2.putText(black_frame, str(pos[0]), (x_min, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                """cv2.putText(black_frame, str(pos[1]), (x_min, y_min + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(black_frame, str(pos[2]), (x_min, y_min + 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(black_frame, str(pos[3]), (x_min, y_min + 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(black_frame, str(pos[4]), (x_min, y_min + 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)"""


                # keypoints
                for i in range(kpts.shape[1]):
                    kpt = kpts[0][i]
                    color_idx = self.kpt_color[i]  # Get the index for the color
                    color = self.pose_palette[color_idx]  # Get the actual color

                    # Draw the keypoint
                    cv2.circle(black_frame, (int(kpt[0]), int(kpt[1])), 3, color.tolist(), -1, lineType=cv2.LINE_AA)

                # Limbs
                for i, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[0][(sk[0] - 1), 0]), int(kpts[0][(sk[0] - 1), 1]))
                    pos2 = (int(kpts[0][(sk[1] - 1), 0]), int(kpts[0][(sk[1] - 1), 1]))

                    color_idx = self.limb_color[i]
                    color = self.pose_palette[color_idx]

                    if pos1[0] % black_frame.shape[1] == 0 or pos1[1] % black_frame.shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % black_frame.shape[1] == 0 or pos2[1] % black_frame.shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue

                    cv2.line(black_frame, pos1, pos2, color.tolist(), 2, lineType=cv2.LINE_AA)


            # Superposer pose_frame sur seg_frame
            seg_frame_with_pose = cv2.addWeighted(seg_frame, 1, black_frame, 1, 0)

            # Afficher les FPS
            self.cv2_txt(seg_frame_with_pose, f'FPS: {round(fps, 2)}', 10, 30, 1)

            # Afficher l'image finale
            cv2.imshow("Detection objets, segmentation et posture", seg_frame_with_pose)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error('Erreur : %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    vision_obj_pers = VisionObjPers()
    rclpy.spin(vision_obj_pers)
    vision_obj_pers.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
