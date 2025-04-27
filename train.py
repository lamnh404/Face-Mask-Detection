from ultralytics import YOLO
from processing_data import process_data


# Process data
process_data()
# Load the model weights from previous training
model = YOLO('yolo11s.pt')

# Continue training without resume parameter
model.train(
    data='data.yaml',
    epochs=300,                # Total training epochs
    imgsz=400,                 # Increased image size for better accuracy (YOLO standard)
    batch=24,                  # Adjusted for larger images
    device='0',                # Using first GPU
    optimizer='AdamW',         # Keeping AdamW optimizer
    lr0=0.001,                 # Lower learning rate for fine-tuning
    lrf=0.01,                  # Final learning rate fraction
    weight_decay=0.0005,       # Regularization to prevent overfitting
    cos_lr=True,               # Cosine learning rate scheduler
    patience=70,               # Early stopping patience
    save=True,                 # Save training results
    cache=True,                # Cache images for faster training
    amp=True,                  # Mixed precision training for speed
    augment=True,              # Enable default augmentations
    mosaic=0.8,                # Slightly reduced mosaic for fine-tuning
    mixup=0.1,                 # Add mixup augmentation
    copy_paste=0.1,            # Add copy-paste augmentation
    hsv_h=0.015,               # Reduced hue augmentation for fine-tuning
    hsv_s=0.7,                 # Increased saturation augmentation
    hsv_v=0.4,                 # Adjusted value augmentation
    fliplr=0.5,                # Standard flip left-right probability
    flipud=0.3,                # Reduced flip up-down probability
)