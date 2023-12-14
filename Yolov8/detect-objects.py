import cv2
from ultralytics import YOLO


# Load the YOLOv8 model
model = YOLO('yolov8m.pt')
#model.fuse()

cap = cv2.VideoCapture("/dev/video6")

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        #print(results[0])

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
        
        


        """# Get all the classes
        Dict[int, str] = results.names

        # From the boxes get the predictions
        results_with_probs: List[Tuple[results, str]] = [(result, classes[result.boxes.cls.numpy()[0]]) for result in results]"""

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()