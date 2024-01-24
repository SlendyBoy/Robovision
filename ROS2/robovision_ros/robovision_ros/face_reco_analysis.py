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
        self.frames = []
        self.frame_count = 0
        # stocker les dernières informations d'attributs
        self.last_age = None
        self.last_genre = None
        self.last_emotion = None
        self.last_ethnie = None
        

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

            face_frame = undistorted_frame

            # face reco
            

            # Save each frame of the video to a list
            self.frame_count += 1
            self.frames.append(frame)

            # Every 5 frames (the default batch size), batch process the list of frames to find faces
            if len(self.frames) == 1:
                batch_of_face_locations = face_recognition.batch_face_locations(self.frames, number_of_times_to_upsample=0, batch_size=1)

                # Stocker les emplacements des visages pour la dernière frame
                last_frame_face_locations = batch_of_face_locations[-1]

                # Analyser uniquement la dernière frame
                last_frame = self.frames[-1]

                for face_location in last_frame_face_locations:
                    top, right, bottom, left = face_location
                    face_image = last_frame[top:bottom, left:right]

                    # Draw a box around the face
                    cv2.rectangle(face_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    if self.frame_count % 100 == 0:

                        # face analysis
                        face_image = face_frame[top:bottom, left:right]

                        try:
                            analysis = DeepFace.analyze(face_image, actions=['age', 'gender', 'race', 'emotion'], detector_backend="opencv", enforce_detection=False, silent=True)
                            result = analysis[0]
                            self.last_age = result['age']
                            self.last_genre = result['dominant_gender']
                            self.last_ethnie = result['dominant_race']
                            self.last_emotion = result['dominant_emotion']
                            

                            
                        except Exception as e:
                            self.get_logger().error('Erreur lors de l\'analyse des attributs du visage: {}'.format(e))

                    if self.frame_count > 99:
                        cv2.putText(face_frame, f"Age : {self.last_age} y.o", (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(face_frame, f"Genre : {self.last_genre}", (left + 6, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(face_frame, f"Ethnie : {self.last_ethnie}", (left + 6, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(face_frame, f"Emotion : {self.last_emotion}", (left + 6, bottom + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)

                    # Clear the frames array to start the next batch
                self.frames = []

            # Afficher les FPS
            self.cv2_txt(face_frame, f'FPS: {round(fps, 2)}', 10, 30, 1)
            self.cv2_txt(face_frame, f'frame: {self.frame_count}', 10, 60, 1)
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
