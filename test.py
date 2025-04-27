from ultralytics import YOLO


# Load the model weights from previous training
model = YOLO('best.pt')
model.eval()
# Process video with optimized parameters
model.predict(
        source='test.mp4',
        conf=0.25,                 # Confidence threshold
        iou=0.5,                  # IoU threshold for NMS
        save=True,                 # Save detection results
        device=0,                   # Use GPU if available (0) or CPU (-1)
    )
