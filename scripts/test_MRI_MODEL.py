from ultralytics import YOLO

# Load your best-trained model
model_path = r'D:/deutschland/BSBI/course content/computer vision/PyCharm projects/project1/runs/pose/train/weights/best.pt'
model = YOLO(model_path)

# Path to the 33 test images
test_folder = r'D:/deutschland/BSBI/course content/computer vision/MRI_DATA/MRI_project/brain_dataset/test_images'

# Run prediction with lower confidence to catch faint tumors
# conf=0.20: Lower threshold to detect tumors with low brightness/clarity
# name='test_low_conf': Saves results in a separate folder for comparison
results = model.predict(
    source=test_folder,
    save=True,
    conf=0.20,
    line_width=2,
    project='D:/deutschland/BSBI/course content/computer vision/PyCharm projects/project1/runs',
    name='test_low_conf',
    exist_ok=True
)

print("Check the folder 'runs/test_low_conf' to see the new detections.")