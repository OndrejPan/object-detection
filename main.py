from ultralytics import YOLO

# Define the data configuration file path
data_config_path = r"C:\Users\panik\PycharmProjects\pythonProject2\Tomato.v1i.yolov8\data.yaml"

# Create a YOLOv8 model instance
model = YOLO('yolov8n.yaml')  # Replace 'yolov8n.yaml' with your model config file if different

# Train the model
results = model.train(data=data_config_path, epochs=100, imgsz=640)