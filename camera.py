import cv2
from ultralytics import YOLO



# Load the model
model = YOLO('best.pt')

video=cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform inference
    results = model(frame,save_dir='results')

    # Process results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes
        confidences = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class IDs

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f'Class: {int(cls)}, Conf: {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv11 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
