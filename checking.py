from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Print the dictionary mapping IDs to class names
print(model.names)