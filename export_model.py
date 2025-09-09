from ultralytics import YOLO

# Load your trained PyTorch model
model = YOLO('models\yolo11n.pt') 

# Export it to TensorRT format
model.export(
    format='engine',  # The format for TensorRT
    half=True,        # Use FP16 precision for speed
    device='cuda'
)
print("Model exported to TensorRT format successfully!")