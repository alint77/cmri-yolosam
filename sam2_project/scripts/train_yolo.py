from ultralytics import YOLO
import argparse
import os

# Load a pretrained YOLOv12m model
# model = YOLO('yolov12m.pt') #Original line

# Train the model
# results = model.train(data='../data/cardiac_mri.yaml', epochs=100, imgsz=640) # Use a small number of epochs for demonstration

# The trained model will be saved automatically by Ultralytics (e.g., best.pt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv12 on cardiac MRI data.")
    parser.add_argument('--weights', type=str, default='yolov12m.pt', help='Path to the pretrained weights file.')
    parser.add_argument('--data', type=str, default='../data/cardiac_mri.yaml', help='Path to the dataset YAML file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training.') # Added batch size argument
    args = parser.parse_args()

    # Construct absolute path to weights file
    if not os.path.isabs(args.weights):
        args.weights = os.path.join(os.getcwd(), args.weights)

    # Construct absolute path to the data YAML file.
    if not os.path.isabs(args.data):
        args.data = os.path.abspath(os.path.join(os.getcwd(), args.data))
        

    model = YOLO(args.weights)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch) # Pass batch size to train()