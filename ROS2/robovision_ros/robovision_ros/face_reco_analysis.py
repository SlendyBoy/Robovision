import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cupy as cp
import tf2_ros

import std_msgs.msg
from deepface import DeepFace
import face_recognition

class FaceRecognitionAnalysis(Node):
    def __init__(self):

        super().__init__('face_reco_analysis') #noeud

        self.last_time_obj = time.time()  # Initialiser last FPS
        self.bridge = CvBridge()
        

        # Intrinsèques
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
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[2][2]

        # Extrinsèques (rotation et translation) de la caméra de profondeur vers la caméra RGB
        self.rotation_matrix = cp.array([
            [0.9999995231628418, -0.0008866226417012513, 0.0003927878278773278], 
            [0.000888532551471144, 0.9999876618385315, -0.004889312665909529], 
            [-0.00038844801019877195, 0.004889659583568573, 0.9999879598617554]
        ])
        self.translation_vector = cp.array([-0.05912087857723236, 0.0001528092980151996, 6.889161159051582e-05])
        
        self.camera_rgb_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_cam,
            10
        )

    def cv2_txt(self, frame, txt, x, y, size):
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), 2, cv2.LINE_AA)

    def undistort_image(self, image):
        return cv2.undistort(image, cp.asnumpy(self.camera_matrix), cp.asnumpy(self.dist_coeffs))
    

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

            # BGR (format openCV) to RGB (format face reco)
            face_frame = undistorted_frame

            # face reco
            face_locations = face_recognition.face_locations(face_frame)


            for face_location in face_locations:

                top, right, bottom, left = face_location
                print(" A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

                # Draw a box around the face
                cv2.rectangle(face_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # face analysis
                face_image = face_frame[top:bottom, left:right]

                try:
                    analysis = DeepFace.analyze(face_image, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                    result = analysis[0]
                    age = result['age']
                    genre = result['dominant_gender']
                    ethnie = result['dominant_race']
                    emotion = result['dominant_emotion']

                    # Maintenant, vous pouvez utiliser ces valeurs pour l'affichage ou le traitement
                    self.get_logger().info(f"Analyse - Âge : {age}, Genre : {genre}, Ethnie : {ethnie}, Émotion : {emotion}")

                    # Draw a label below the face
                    cv2.putText(face_frame, f"Age : {age} ans", (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(face_frame, f"Genre : {genre}", (left + 6, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(face_frame, f"Ethnie : {ethnie}", (left + 6, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(face_frame, f"Emotion : {emotion}", (left + 6, bottom + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                except Exception as e:
                    self.get_logger().error('Erreur lors de l\'analyse des attributs du visage: {}'.format(e))

            # Afficher les FPS
            self.cv2_txt(face_frame, f'FPS: {round(fps, 2)}', 10, 30, 1)
            # Afficher l'image finale
            cv2.imshow("Face analysis", face_frame)
            cv2.waitKey(1)


        except Exception as e:
            self.get_logger().error('Erreur : %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    face_reco_analysis = FaceRecognitionAnalysis()
    rclpy.spin(face_reco_analysis)
    face_reco_analysis.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
