import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define image dimensions and number of channels - RGB for the CNN
img_height = 128
img_width = 128
num_channels = 3

# Load or create the ball detection CNN model
if os.path.exists('ball_detection_model.h5'):
    ball_model = tf.keras.models.load_model('ball_detection_model.h5')
else:
    # Define the model architecture
    ball_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='linear'),  # Use 'linear' activation for bounding box coordinates
    ])
    ball_model.compile(optimizer='adam', loss='mean_squared_error') #error function is MSE 
    ball_model.save('ball_detection_model.h5')

# Load and preprocess training data
train_data_directory = 'D:/flutter/soccer_ball_detection/soccer_exercise/train'
train_annotations = pd.read_csv(os.path.join(train_data_directory, 'annotations.csv'))

image_data = []
targets = []

for index, row in train_annotations.iterrows():
    image_file_name = os.path.join(train_data_directory, row['filename'])
    
    image = Image.open(image_file_name).resize((img_width, img_height))
    image = np.array(image) / 255.0
    image_data.append(image)

    xmin = row['xmin'] / img_width
    ymin = row['ymin'] / img_height
    xmax = row['xmax'] / img_width
    ymax = row['ymax'] / img_height
    
    target = [xmin, ymin, xmax, ymax]
    targets.append(target)

image_data = np.array(image_data)
targets = np.array(targets)

# Split data into training and validation sets
ratio_of_split = 0.8
index_of_split = int(len(image_data) * ratio_of_split)

training_images = image_data[:index_of_split]
training_targets = targets[:index_of_split]
validation_images = image_data[index_of_split:]
validation_targets = targets[index_of_split:]

# Custom loss function
class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Calculate mean squared error loss for bounding box coordinates
        return tf.reduce_mean(tf.square(y_true - y_pred))

# Custom training loop
optimizer = tf.keras.optimizers.Adam()
loss_object = CustomLoss()

training_losses = []
batch_size = 16  # Adjust as needed

for epoch in range(10): #increasing the epochs leads to poorer predictions in the model and also risks overfitting the model
    epoch_losses = []
    for i in range(0, len(training_images), batch_size):
        images_batch = training_images[i:i+batch_size]
        targets_batch = training_targets[i:i+batch_size]

        # Make predictions using the model
        with tf.GradientTape() as tape:
            predictions = ball_model(images_batch, training=True)
            
            loss_value = loss_object(targets_batch, predictions)

        grads = tape.gradient(loss_value, ball_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, ball_model.trainable_variables))

        epoch_losses.append(loss_value.numpy())
        
        # Print predictions and actual targets for a batch during training
        if i == 0 and epoch % 2 == 0:
            print("Predictions:", predictions)
            print("Actual Targets:", targets_batch)

    epoch_loss = np.mean(epoch_losses)
    training_losses.append(epoch_loss)

    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss)) #example: Epoch 009: Loss: 1.134

# Plot training losses
plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Save the trained model under a different name
new_model_name = 'ball_detection_model_trained.h5'
ball_model.save(new_model_name)
print("Trained model saved as:", new_model_name)
