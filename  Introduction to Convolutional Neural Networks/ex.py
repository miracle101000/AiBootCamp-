import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

# # Load Dataset
# transform = transforms.ToTensor()
# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)


# # Visualize sample images
# fig, axes = plt.subplots(1, 5, figsize=(12, 3))
# for i in range(5):
#     image, label =  train_dataset[i]
#     axes[i].imshow(image.permute(1, 2, 0))
#     axes[i].axis('off')
#     axes[i].set_title(f"Label: {label}")
# plt.show()    

# # Display pixel values for the first image
# image, label = train_dataset[0]
# print(f"Label: {label}")
# print(f"Image Shape: {image.shape}")
# print("Pixel Values:")
# print(image)


import tensorflow as tf

#Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Tensorflow CNN Model is ready")

import torch.nn as nn

#Define a simple  CNN Model
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)  # No activation here
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 15 * 15, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU after convolution
        x = self.pool(x)  # Apply max pooling
        x = x.view(-1, 32 * 15 * 15)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = self.fc2(x)  # Output layer (no activation because we'll use CrossEntropyLoss)
        return x

print("PyTorch CNN model Ready")  
        
        