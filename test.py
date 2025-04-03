import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from model.model_functional import YOLOv3
from utils.utils import get_classes, get_anchors, preprocess_input
from configs import *

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the box with highest score
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        
        # Get IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id:box_id+1, :], boxes[sorted_indices[1:], :])
        
        # Remove boxes with IoU over threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return keep_boxes

def compute_iou(box, boxes):
    # Calculate intersection areas
    x1 = np.maximum(box[0, 0], boxes[:, 0])
    y1 = np.maximum(box[0, 1], boxes[:, 1])
    x2 = np.minimum(box[0, 2], boxes[:, 2])
    y2 = np.minimum(box[0, 3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    box_area = (box[0, 2] - box[0, 0]) * (box[0, 3] - box[0, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = box_area + boxes_area - intersection
    
    return intersection / union

def detect_image(image_path, model_path, classes_path, anchors_path, input_shape=(416, 416)):
    print(f'Processing image: {image_path}')
    
    # Load classes and anchors
    class_names, num_classes = get_classes(classes_path)
    print(f'Loaded {num_classes} classes: {class_names}')
    
    anchors, num_anchors = get_anchors(anchors_path)
    print(f'Loaded {num_anchors} anchors')
    
    # Create YOLOv3 model with correct input shape (height, width, channels)
    model = YOLOv3(input_shape=(input_shape[0], input_shape[1], 3), num_classes=num_classes)
    
    # Load trained weights
    print(f'Loading weights from {model_path}')
    model.load_weights(model_path)
    
    # Read and preprocess image
    print('Reading and preprocessing image...')
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_shape = np.array(np.shape(image)[0:2])
    print(f'Original image shape: {image_shape}')
    
    # Resize image
    input_shape = np.array(input_shape)
    image_data = image.resize(input_shape)
    image_data = np.array(image_data, dtype='float32')
    print(f'Resized image shape: {image_data.shape}')
    
    # Preprocess image to match expected shape (batch_size, height, width, channels)
    image_data = preprocess_input(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    print(f'Preprocessed image shape: {image_data.shape}')
    
    # Get predictions
    print('Running inference...')
    output = model.predict(image_data)
    print(f'Model output shape: {[o.shape for o in output]}')
    
    # Process predictions
    boxes = []
    scores = []
    classes = []
    
    # Process each scale output
    for i, out in enumerate(output):
        # Reshape output to (batch_size, grid_size * grid_size * num_anchors, num_classes + 5)
        batch_size, grid_h, grid_w, _ = out.shape
        out = np.reshape(out, (batch_size, grid_h * grid_w * 3, -1))
        
        # Get boxes, scores, and classes
        box_xy = out[..., :2]
        box_wh = out[..., 2:4]
        box_confidence = out[..., 4:5]
        box_class_probs = out[..., 5:]
        
        # Convert box coordinates to absolute coordinates
        box_xy = box_xy / np.array([grid_w, grid_h])
        box_wh = box_wh / np.array([input_shape[1], input_shape[0]])
        
        # Convert to corner coordinates
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        _boxes = np.concatenate([box_mins, box_maxes], axis=-1)
        
        # Get scores and classes
        _scores = np.max(box_confidence * box_class_probs, axis=-1)
        _classes = np.argmax(box_class_probs, axis=-1)
        
        boxes.append(_boxes)
        scores.append(_scores)
        classes.append(_classes)
    
    # Combine predictions from all scales
    boxes = np.concatenate(boxes, axis=1)
    scores = np.concatenate(scores, axis=1)
    classes = np.concatenate(classes, axis=1)
    
    # Filter predictions based on confidence threshold
    confidence_threshold = 0.85  # Increased from 0.7
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    print(f'Found {len(boxes)} detections above confidence threshold {confidence_threshold}')
    
    # Convert boxes to image coordinates
    boxes[..., [0, 2]] *= image_shape[1]  # scale x coordinates
    boxes[..., [1, 3]] *= image_shape[0]  # scale y coordinates
    
    # Apply Non-Maximum Suppression
    keep_indices = nms(boxes, scores, iou_threshold=0.3)  # Decreased from 0.5
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    classes = classes[keep_indices]
    print(f'After NMS: {len(boxes)} detections')
    
    # Draw boxes on image
    image = np.array(image)
    
    # Define colors for each class (BGR format for OpenCV)
    colors = {
        0: (0, 255, 0),    # Green for T_or_L_shape_mark
        1: (255, 0, 0),    # Blue for Right_angle_head
        2: (0, 0, 255),    # Red for Acute_angle_head
        3: (0, 255, 255)   # Yellow for Obtuse_angle_head
    }
    
    # Count detections per class
    class_counts = {}
    for cls in classes:
        cls = int(cls)
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nDetections by class:")
    for cls, count in class_counts.items():
        print(f"{class_names[cls]}: {count}")
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        cls = int(cls)
        color = colors[cls]
        
        # Draw thicker rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Add filled background for label
        label = f'{class_names[cls]} {score:.2f}'
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
        
        # Add white text label
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 30
    for cls, color in colors.items():
        if cls in class_counts:
            legend_text = f'{class_names[cls]}: {class_counts[cls]}'
            cv2.putText(image, legend_text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)
            legend_y += 30
    
    # Save output image
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')
    
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f'Detection results saved to {output_path}')

def main():
    # Paths
    model_path = 'logs/loss_2025_04_03_07_47_44ep030-loss502.928-val_loss1490.348.h5'  # Use the latest trained weights
    classes_path = PATH_CLASSES
    anchors_path = PATH_ANCHORS
    
    # Test on a single image
    image_path = 'data/demo/train/20160725-7-158.jpg'  # Use your test image
    detect_image(image_path, model_path, classes_path, anchors_path)

if __name__ == '__main__':
    main() 