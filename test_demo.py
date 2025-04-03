import cv2
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from model.model_functional import YOLOv3
from utils.utils import preprocess_input, resize_image, convert2rgb, get_anchors
from utils.utils_bbox import DecodeBox, yolo_correct_boxes
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

def preprocess_image(image_path, input_shape=(416, 416)):
    """Preprocess image for YOLOv3 model."""
    # Load and convert image to RGB
    image = Image.open(image_path)
    image = convert2rgb(image)
    
    # Store original size
    original_size = image.size
    
    # Resize image
    image_data = resize_image(image, input_shape, True)
    
    # Convert to numpy array and normalize
    image_data = np.array(image_data, dtype=np.float32)
    image_data = preprocess_input(image_data)
    
    # Add batch dimension
    image_data = np.expand_dims(image_data, axis=0)
    
    return image_data, image, original_size

def process_predictions(outputs, anchors, num_classes, input_shape, image_shape, conf_thresh=0.3):
    """Process raw YOLO outputs into bounding boxes."""
    boxes = []
    scores = []
    classes = []
    
    # Process each output feature map
    for output, anchor_group in zip(outputs, [anchors[6:9], anchors[3:6], anchors[0:3]]):
        # Get grid size
        grid_h, grid_w = output.shape[1:3]
        
        # Reshape output
        output = np.reshape(output, (1, grid_h, grid_w, 3, 5 + num_classes))
        
        # Process each grid cell
        for i in range(grid_h):
            for j in range(grid_w):
                for b in range(3):  # 3 anchors per cell
                    # Get confidence score
                    confidence = output[0, i, j, b, 4]
                    
                    if confidence > conf_thresh:
                        # Get class probabilities
                        class_probs = output[0, i, j, b, 5:]
                        class_id = np.argmax(class_probs)
                        class_score = class_probs[class_id]
                        
                        # Calculate final score
                        score = confidence * class_score
                        
                        if score > conf_thresh:
                            # Get box coordinates
                            x = (output[0, i, j, b, 0] + j) / grid_w
                            y = (output[0, i, j, b, 1] + i) / grid_h
                            w = np.exp(output[0, i, j, b, 2]) * anchor_group[b][0] / input_shape[1]
                            h = np.exp(output[0, i, j, b, 3]) * anchor_group[b][1] / input_shape[0]
                            
                            # Convert to corner coordinates
                            x1 = int((x - w/2) * image_shape[1])
                            y1 = int((y - h/2) * image_shape[0])
                            x2 = int((x + w/2) * image_shape[1])
                            y2 = int((y + h/2) * image_shape[0])
                            
                            boxes.append([x1, y1, x2, y2])
                            scores.append(score)
                            classes.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(classes)

def draw_boxes(image, boxes, scores, classes, class_names=None):
    """Draw bounding boxes on the image."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[i]
        class_id = classes[i]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Parking Slot: {score:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    # Model parameters
    input_shape = (416, 416)
    num_classes = 1  # Only detecting parking slots
    anchors_path = 'model_data/yolo_anchors.txt'
    weights_path = 'model_data/yolov3_weights.h5'  # Path to your trained weights
    
    # Get anchors
    anchors, num_anchors = get_anchors(anchors_path)
    
    # Create model
    model = YOLOv3(input_shape + (3,), num_classes)
    
    # Load weights
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}")
        print("Please train the model first or provide the correct weights path")
        return
    
    # Load image
    image_path = 'data/demo/train/20160725-7-158.jpg'
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        exit(1)
    
    # Preprocess image
    image_data, original_image, original_size = preprocess_image(image_path, input_shape)
    
    # Get predictions
    outputs = model.predict(image_data)
    
    # Process predictions with lower confidence threshold
    boxes, scores, classes = process_predictions(
        outputs,
        anchors,
        num_classes,
        input_shape,
        (original_size[1], original_size[0]),  # height, width
        conf_thresh=0.3  # Lower threshold for more detections
    )
    
    # Convert image for display
    display_image = np.array(original_image)
    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
    
    # Draw boxes
    result_image = draw_boxes(display_image, boxes, scores, classes)
    
    # Display result
    cv2.imshow('Parking Slot Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 