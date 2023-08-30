import os
import numpy as np
import pandas as pd 
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define image dimensions and number of channels - RGB for the CNN
img_height = 128
img_width = 128
num_channels = 3

# Load and preprocess training data
train_data_directory = 'D:/flutter/soccer_ball_detection/soccer_exercise/train'  # Replace with the actual path
train_annotations = pd.read_csv(os.path.join(train_data_directory, 'annotations.csv'))

image_data = []
labels = []

#extract images and labels from the training data and store them in image_data[] and labels[]. 
#the image is resized to 128*128 to match our model's specifications. 
#the label is set to 1 if the class is football. Otherwise, the label is set to 0 (for referee and player classes).
for index, row in train_annotations.iterrows():
    image_file_name = os.path.join(train_data_directory, row['filename'])
    
    image = Image.open(image_file_name).resize((img_width, img_height))
    image = np.array(image) / 255.0
    image_data.append(image)

    label = 1 if 'class' in row and row['class'] == 'football' else 0
    labels.append(label)

image_data = np.array(image_data)
labels = np.array(labels)

# Split data into training and validation sets
#0.8 (80%) of training data is used for training and 20% for validation
ratio_of_split = 0.8
#Multiply the total number of images by the ratio_of_split and convert the result to an integer. 
#This index will mark the boundary between the training and validation sets.
index_of_split = int(len(image_data) * ratio_of_split)

#Extract training images and labels
#Extract the images from the beginning of the image_data array up to the index_of_split. These images will be used for training.
training_images = image_data[:index_of_split]
#Extract the labels from the beginning of the image_data array up to the index_of_split. These labels will be used for training.
training_labels = labels[:index_of_split]
#Extract the images from the index_of_split to the end of the image_data array. These images will be used for validation.
validation_images = image_data[index_of_split:]
#Extract the labels from the index_of_split to the end of the image_data array. These labels will be used for validation.
validation_labels = labels[index_of_split:]

# Define the ball classifier model
#It uses 32 filters of size 3x3 to extract features from the input image.
#ReLU activation function introduces non-linearity to the model, allowing it to learn complex patterns in the data.
#Max-pooling is applied to the output of the previous convolutional layer. It reduces the spatial dimensions of the 
#feature maps, helping to decrease the number of parameters in the model. 2*2 means we are picking the highest value 
#in each 2x2 region
#Conv2D(64, (3,3), activation='relu') further extracts more complex features from the reduced-size feature maps.
#Another max-pooling layer is used to reduce/ downsample the feature maps again.
#Conv2D(128, (3, 3), activation='relu') performs another round of convolution with an increased number of filters to capture even higher-level features.
#Another max-pooling layer is used to further reduce spatial dimensions.
#Flatten() is used to flatten the 3D feature maps into a 1D vector.
#Dense(128, activation='relu') is a fully connected (dense) layer has 128 neurons. It takes the flattened feature vector as input and performs a linear 
#transformation followed by the ReLU activation function.
#Dropout is a regularization technique that randomly sets a fraction of the input units to zero during training.
#It helps prevent overfitting by reducing the reliance on any single neuron.
#Dense layer with 1 unit produces the binary classification output (0 or 1) indicating whether the input image contains a soccer ball or not.
#The sigmoid activation function squashes the output to the range [0, 1], representing the probability of the positive class (soccer ball presence).
ball_classifier = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),  # Binary classification output
])

# Compile the model using adam optimizer, binary crossentropy and accuracy metrics
ball_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Custom training loop
optimizer = tf.keras.optimizers.Adam() #Adam stands for "Adaptive Moment Estimation" and is a popular optimization method for training neural networks.
#It is essential for updating the neural network's weights based on the gradients of the loss function during the training process.
loss_object = tf.keras.losses.BinaryCrossentropy() #Cross-entropy loss, or log loss, measures the performance of a 
#classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted 
#probability diverges from the actual label.

batch_size = 16
epochs = 10 #increasing the epochs leads to poorer predictions in the model and also risks overfitting the model

#custom training loop. It iterates through epochs and within each epoch, iterates through the training data in batches. 
#The model's predictions are compared to the ground truth labels, and the loss is calculated. 
#The gradients are computed and used to update the model's weights.
for epoch in range(epochs):
    epoch_losses = []
    for i in range(0, len(training_images), batch_size):
        images_batch = training_images[i:i+batch_size]
        labels_batch = training_labels[i:i+batch_size]

        #tf.GradientTape() context traces operations that involve tensors and calculate gradients with respect to those tensors.
        #It's used for automatic differentiation, a key component of backpropagation in neural network training.
        #tape refers to the gradient tape context that was previously created using tf.GradientTape(). It records operations to compute gradients.
        with tf.GradientTape() as tape:
            #Call the neural network model, ball_classifier, with the images_batch as input, and set training=True.
            #Compute forward pass of the model, generating predictions for the given batch of input images.
            predictions = ball_classifier(images_batch, training=True)
            #loss function is computed using the model's predictions and the ground truth labels for the current batch.
            #tf.expand_dims(labels_batch, axis=-1) is used to expand the dimensions of labels_batch to match the shape of predictions. 
            loss_value = loss_object(tf.expand_dims(labels_batch, axis=-1), predictions)

        #Calculate the gradients of the loss function (loss_value) with respect to the trainable variables of the neural network model.
        #tape refers to the gradient tape context that was previously created using tf.GradientTape(). It records operations to compute gradients.
        #loss_value is the computed loss value associated with the current batch of data.
        #ball_classifier.trainable_variables represents the list of all trainable parameters (weights and biases) in the neural network model.
        grads = tape.gradient(loss_value, ball_classifier.trainable_variables)
        #Apply the calculated gradients to update the model's trainable variables (weights and biases) using the Adam optimizer.
        #zip(grads, model.trainable_variables) pairs each gradient with its corresponding trainable variable.
        optimizer.apply_gradients(zip(grads, ball_classifier.trainable_variables))

        epoch_losses.append(loss_value.numpy())

    epoch_loss = np.mean(epoch_losses)
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss)) #example: Epoch 009: Loss: 0.261

# Save the trained classifier model
ball_classifier.save('ball_classifier_model.h5')
print("Ball classifier model saved as: ball_classifier_model.h5")
