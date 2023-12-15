import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time
from cv_bridge import CvBridge
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

        cv2.namedWindow("Detection objets et posture", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        
        self.subscription_camera_rgb = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_cam,
            10
        )

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
        

    def listener_cam(self, msg):

        try:

            current_time = time.time()
            elapsed_time = current_time - self.last_time_obj
            if elapsed_time > 0:
                fps = 1.0 / elapsed_time
            self.last_time_obj = current_time

            # Convertir le message ROS en une image OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # YOLOv8 obj inference
            results = model_obj(frame, conf=0.5)

            results_np = results[0].cpu().numpy()

            for result in results_np:
                boxes = result.boxes
                cls = boxes.cls
                print(boxes)
                print(cls)

                #print("=================================")

            """boxes = results[0].boxes.xywh.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
            for box, cls in zip(boxes, clss):
                x, y, w, h = box
                label = str(names[int(cls)])
                print(x, y, w, h)
                print(label)"""


            datas_frame = results_np.plot()

            

            #cv2.imshow("Detection objets", datas_frame)

            # YOLOv8 pose inference
            results_pose = model_pose(datas_frame, conf=0.5)

            pose_frame = results_pose[0].plot()

            # Display the annotated frame
            self.cv2_txt(pose_frame, f'FPS: {round(fps, 2)}', 10, 30, 1)
            cv2.imshow("Detection objets et posture", pose_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error('Erreur lors de la conversion de l\'image ROS en image OpenCV: %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    vision_obj_pers = VisionObjPers()
    rclpy.spin(vision_obj_pers)
    vision_obj_pers.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
