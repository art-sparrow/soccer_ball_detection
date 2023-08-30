import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set TensorFlow to only display error messages
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tensorflow as tf

# Load the trained ball classifier model
ball_classifier_model_path = 'ball_classifier_model.h5'
ball_classifier = tf.keras.models.load_model(ball_classifier_model_path)

# Load the trained ball detection model
ball_detection_model_path = 'ball_detection_model_trained.h5'
ball_detection_model = tf.keras.models.load_model(ball_detection_model_path, compile=False)

# Define image dimensions and number of channels - RGB for the CNN
img_height = 128
img_width = 128
num_channels = 3

# Define the width of the bounding box outline
box_outline_width = 3

# Load test data annotations
test_data_directory = 'D:/flutter/soccer_ball_detection/soccer_exercise/test'  # Replace with the actual path
test_annotations = pd.read_csv(os.path.join(test_data_directory, 'annotations.csv'))

# Create a directory for storing prediction images
output_directory = 'D:/flutter/soccer_ball_detection/soccer_exercise/predictions'
os.makedirs(output_directory, exist_ok=True)

# Initialize variables for counting correct predictions
correct_predictions = 0
total_predictions = len(test_annotations)

#calculate accuracy using intersection over union to check the number of times the 
#predicted coordinates overlap with true ground values
def is_prediction_correct(prediction, true_xmin, true_ymin, true_xmax, true_ymax):
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = prediction
    
    # Calculate Intersection over Union (IoU)
    intersection_xmin = max(pred_xmin, true_xmin)
    intersection_ymin = max(pred_ymin, true_ymin)
    intersection_xmax = min(pred_xmax, true_xmax)
    intersection_ymax = min(pred_ymax, true_ymax)
    
    intersection_area = max(0, intersection_xmax - intersection_xmin) * max(0, intersection_ymax - intersection_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    
    iou = intersection_area / (pred_area + true_area - intersection_area)
    
    # Define a threshold for IoU to consider the prediction correct
    iou_threshold = 0 #increasing it means you get more accurate results, but also locks out some of the predictions
    
    return iou >= iou_threshold

# Iterate through test images and annotations
for index, row in test_annotations.iterrows():
    image_file_name = os.path.join(test_data_directory, row['filename'])
    image_original = Image.open(image_file_name) #store image with original dimensions
    image = image_original.resize((img_width, img_height)) #convert the image to 128*128 for the models
    image = np.array(image) / 255.0

    # Reshape the image for the classifier
    input_image = np.expand_dims(image, axis=0)

    # Use the classifier to predict whether the image contains a ball
    predicted_class = ball_classifier.predict(input_image)[0][0]
    print("Predicted Class:", predicted_class)

    if predicted_class > 0.0 and row['class'] == 'football':  # Threshold for ball detection
        # Use the detection model for inference
        prediction = ball_detection_model(input_image)[0]
        print("Detection Prediction:", prediction)

        # Extract predicted box coordinates
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = prediction

        # Convert to 128x128 dimensions using floating-point values
        pred_xmin_128 = round(float(pred_xmin) * 128)
        pred_ymin_128 = round(float(pred_ymin) * 128)
        pred_xmax_128 = round(float(pred_xmax) * 128)
        pred_ymax_128 = round(float(pred_ymax) * 128)
        print("Converted to original dimensions:", pred_xmin_128, pred_ymin_128, pred_xmax_128, pred_ymax_128)

        # Draw red rectangle around football position on the original image
        img_with_box = image_original.copy()
        draw = ImageDraw.Draw(img_with_box)
        draw.rectangle(
            [(pred_xmin_128, pred_ymin_128), (pred_xmax_128, pred_ymax_128)],
            outline='red',
            width=box_outline_width
        )

        # Save the image with prediction
        output_path = os.path.join(output_directory, f'prediction_{index}.png')
        img_with_box.save(output_path)

        # Increment correct_predictions if the prediction is accurate
        if is_prediction_correct(prediction, row['xmin'], row['ymin'], row['xmax'], row['ymax']):
            correct_predictions += 1

# Calculate accuracy as a percentage
accuracy_percentage = (correct_predictions / total_predictions) * 100

# Print the accuracy
print(f"Accuracy: {accuracy_percentage:.2f}%")
