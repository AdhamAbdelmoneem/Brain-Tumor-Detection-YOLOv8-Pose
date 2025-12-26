from ultralytics import YOLO

# Load the pose model
model = YOLO("yolov8n-pose.pt")

# Start training with optimized settings for CPU
model.train(
    data=r'D:\deutschland\BSBI\course content\computer vision\MRI_DATA\MRI_project\brain_dataset\data.yaml',
    epochs=100,
    imgsz=640,
    device='cpu',
    batch=4,        # Small batch size to save CPU memory
    workers=0,      # Required for Windows to avoid multiprocessing errors
    exist_ok=True   # Overwrite existing experiment folders if needed
)