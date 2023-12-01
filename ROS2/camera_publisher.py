import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'test_camera_image', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.cap = cv2.VideoCapture(0)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            ros_image = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.cap.release()
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
